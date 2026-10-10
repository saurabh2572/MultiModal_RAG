"""Load vector and keyword stores, fuse results, and expand page context."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .keyword_index import KeywordIndex, file_sha256


class ChunkStore:
    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        self.sha256 = file_sha256(self.path)
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_page: dict[tuple[str, int, str], list[str]] = defaultdict(list)
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk_id = chunk["id"]
                if chunk_id in self.by_id:
                    raise ValueError(f"Duplicate chunk ID: {chunk_id}")
                self.by_id[chunk_id] = chunk
                metadata = chunk["metadata"]
                key = (metadata["source"], metadata["page"], metadata.get("section", ""))
                self.by_page[key].append(chunk_id)

    def expand(self, hits: list[dict[str, Any]], *, max_sources: int = 5,
               max_chars: int = 18000) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        total_chars = 0
        for hit in hits[:max_sources]:
            chunk = self.by_id[hit["chunk_id"]]
            meta = chunk["metadata"]
            key = (meta["source"], meta["page"], meta.get("section", ""))
            peers = self.by_page[key]
            position = peers.index(hit["chunk_id"])
            nearby = [self.by_id[peer_id] for peer_id in peers[max(0, position - 1):position + 2]]
            text = "\n\n".join(item["text"] for item in nearby)
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            text = text[:remaining]
            total_chars += len(text)
            evidence.append({
                "citation": len(evidence) + 1,
                "chunk_id": hit["chunk_id"],
                "source": meta["source"],
                "page": meta["page"],
                "page_label": meta.get("page_label", str(meta["page"])),
                "section": meta.get("section", ""),
                "modality": meta["modality"],
                "image_path": meta.get("image_path"),
                "text": text,
            })
        return evidence


class DenseRetriever:
    def __init__(self, db_dir: Path, *, expected_sha256: str, device: str = "cpu"):
        db_dir = db_dir.resolve(strict=True)
        manifest = json.loads((db_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != expected_sha256:
            raise ValueError("Vector DB was built from different chunks.jsonl")
        if manifest.get("max_chunks") is not None:
            raise ValueError("Online retrieval requires a complete vector DB, not a --max-chunks trial")
        self.device = device
        self.descriptions = manifest["spaces"]
        self.indexes: dict[str, Any] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.text_models: dict[str, Any] = {}
        self.clip_model = None
        self.clip_tokenizer = None
        self._model_lock = threading.RLock()
        for space, description in self.descriptions.items():
            folder = db_dir / space
            index = faiss.read_index(str(folder / "index.faiss"))
            records = [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
            if index.ntotal != len(records) or index.ntotal != description["count"]:
                raise ValueError(f"Vector index and records differ for {space}")
            self.indexes[space] = index
            self.records[space] = records

    def _query_vector(self, space: str, query: str) -> np.ndarray:
        description = self.descriptions[space]
        if space == "image_visual":
            import open_clip
            import torch

            if self.clip_model is None:
                self.clip_model, _, _ = open_clip.create_model_and_transforms(
                    description["model"], pretrained=description["pretrained"], device=self.device,
                )
                self.clip_model.eval()
                self.clip_tokenizer = open_clip.get_tokenizer(description["model"])
            with torch.inference_mode():
                features = self.clip_model.encode_text(self.clip_tokenizer([query]).to(self.device))
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vector = features.float().cpu().numpy()
        else:
            from sentence_transformers import SentenceTransformer

            model_name = description["model"]
            if model_name not in self.text_models:
                self.text_models[model_name] = SentenceTransformer(model_name, device=self.device, local_files_only=True)
            vector = self.text_models[model_name].encode([query], convert_to_numpy=True, normalize_embeddings=True)
        vector = np.ascontiguousarray(vector, dtype="float32")
        faiss.normalize_L2(vector)
        return vector

    def search(self, query: str, *, include_visual: bool, per_space: int = 8, limit: int = 20) -> list[dict[str, Any]]:
        # Lazy model loading and inference share mutable model state.
        with self._model_lock:
            return self._search(query, include_visual=include_visual, per_space=per_space, limit=limit)

    def _search(self, query: str, *, include_visual: bool, per_space: int, limit: int) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for space in ("text", "table", "image_text", "image_visual"):
            if space not in self.indexes or (space == "image_visual" and not include_visual):
                continue
            vector = self._query_vector(space, query)
            index = self.indexes[space]
            if vector.shape[1] != index.d:
                raise ValueError(f"Query/index dimension mismatch for {space}")
            scores, positions = index.search(vector, min(per_space, index.ntotal))
            for rank, (score, position) in enumerate(zip(scores[0], positions[0]), 1):
                if position < 0:
                    continue
                chunk_id = self.records[space][int(position)]["chunk_id"]
                item = fused.setdefault(chunk_id, {"chunk_id": chunk_id, "score": 0.0, "channels": []})
                item["score"] += 1 / (60 + rank)
                item["channels"].append({"space": space, "rank": rank, "cosine": float(score)})
        return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]


def fuse_hits(dense: list[dict[str, Any]], keyword: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for channel, hits in (("dense", dense), ("bm25", keyword)):
        for rank, hit in enumerate(hits, 1):
            chunk_id = hit["chunk_id"]
            item = fused.setdefault(chunk_id, {"chunk_id": chunk_id, "score": 0.0, "channels": []})
            item["score"] += 1 / (60 + rank)
            item["channels"].append(channel)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._model_lock = threading.RLock()

    def rerank(self, query: str, hits: list[dict[str, Any]], chunks: ChunkStore,
               *, limit: int = 5) -> list[dict[str, Any]]:
        with self._model_lock:
            return self._rerank(query, hits, chunks, limit=limit)

    def _rerank(self, query: str, hits: list[dict[str, Any]], chunks: ChunkStore,
                *, limit: int) -> list[dict[str, Any]]:
        if not hits:
            return []
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
        pairs = [(query, chunks.by_id[hit["chunk_id"]]["text"][:2000]) for hit in hits]
        scores = self.model.predict(pairs)
        ranked = [dict(hit, rerank_score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:limit]


def load_retrievers(chunks_path: Path, vector_db_dir: Path, keyword_db_path: Path,
                    rerank_model: str, device: str) -> tuple[ChunkStore, DenseRetriever, KeywordIndex, Reranker]:
    chunks = ChunkStore(chunks_path)
    return (
        chunks,
        DenseRetriever(vector_db_dir, expected_sha256=chunks.sha256, device=device),
        KeywordIndex(keyword_db_path, expected_sha256=chunks.sha256),
        Reranker(rerank_model, device=device),
    )
