"""Build separate FAISS indexes for document text, tables, and images.

Text, table, and image-description vectors use BGE. Pixel vectors use CLIP.
Their dimensions and score distributions differ, so each space has its own index.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np


TEXT_SPACES = ("text", "table", "image_text")
IMAGE_SPACE = "image_visual"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(chunks_path: Path, max_chunks: int | None) -> dict[str, list[dict[str, Any]]]:
    spaces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    with chunks_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("id")
            metadata = chunk.get("metadata", {})
            modality = metadata.get("modality")
            text = chunk.get("text", "").strip()
            if not chunk_id or chunk_id in seen_ids or modality not in {"text", "table", "image"}:
                raise ValueError(f"Invalid or duplicate chunk at line {line_number}")
            seen_ids.add(chunk_id)
            if modality in {"text", "table"}:
                if not text:
                    raise ValueError(f"Empty {modality} chunk at line {line_number}")
                spaces[modality].append(chunk)
            else:
                image_path = metadata.get("image_path")
                if not image_path or not Path(image_path).is_file():
                    raise FileNotFoundError(f"Image missing for chunk {chunk_id}: {image_path}")
                spaces[IMAGE_SPACE].append(chunk)
                if "OCR:" in text or "Caption:" in text:
                    spaces["image_text"].append(chunk)
            if max_chunks is not None and len(seen_ids) >= max_chunks:
                break
    if not seen_ids:
        raise ValueError(f"No chunks in {chunks_path}")
    return spaces


def _embedding_text(chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    parts = [metadata.get("section"), metadata.get("subsection"), chunk["text"]]
    return "\n".join(part for part in parts if part)


def _record(chunk: dict[str, Any], vector_id: int) -> dict[str, Any]:
    return {
        "vector_id": vector_id,
        "chunk_id": chunk["id"],
        "element_id": chunk.get("element_id"),
        "text": chunk["text"],
        "metadata": chunk["metadata"],
    }


def _write_text_space(
    root: Path,
    name: str,
    chunks: list[dict[str, Any]],
    model: Any,
    model_name: str,
    batch_size: int,
) -> dict[str, Any]:
    space_dir = root / name
    space_dir.mkdir()
    index = None
    with (space_dir / "records.jsonl").open("w", encoding="utf-8") as records_file:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = np.asarray(
                model.encode(
                    [_embedding_text(chunk) for chunk in batch],
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype="float32",
            )
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError(f"Unexpected embedding shape for {name}: {vectors.shape}")
            vectors = np.ascontiguousarray(vectors)
            faiss.normalize_L2(vectors)
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            for offset, chunk in enumerate(batch):
                records_file.write(json.dumps(_record(chunk, start + offset), ensure_ascii=False) + "\n")
            print(f"{name}: {start + len(batch)}/{len(chunks)} vectors", flush=True)
    assert index is not None
    faiss.write_index(index, str(space_dir / "index.faiss"))
    return {"count": index.ntotal, "dimension": index.d, "model": model_name, "metric": "cosine", "records": f"{name}/records.jsonl"}


def _write_image_space(
    root: Path,
    chunks: list[dict[str, Any]],
    model_name: str,
    pretrained: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import open_clip
    import torch
    from PIL import Image

    space_dir = root / IMAGE_SPACE
    space_dir.mkdir()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    index = None
    with (space_dir / "records.jsonl").open("w", encoding="utf-8") as records_file:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            tensors = []
            for chunk in batch:
                with Image.open(chunk["metadata"]["image_path"]) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            images = torch.stack(tensors).to(device)
            with torch.inference_mode():
                features = model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors = np.ascontiguousarray(features.float().cpu().numpy(), dtype="float32")
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            for offset, chunk in enumerate(batch):
                records_file.write(json.dumps(_record(chunk, start + offset), ensure_ascii=False) + "\n")
            print(f"{IMAGE_SPACE}: {start + len(batch)}/{len(chunks)} vectors", flush=True)
    assert index is not None
    faiss.write_index(index, str(space_dir / "index.faiss"))
    return {
        "count": index.ntotal,
        "dimension": index.d,
        "model": model_name,
        "pretrained": pretrained,
        "metric": "cosine",
        "records": f"{IMAGE_SPACE}/records.jsonl",
    }


def build_vector_db(
    chunks_path: Path,
    output_dir: Path,
    *,
    text_model: str = "BAAI/bge-small-en-v1.5",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    device: str = "cpu",
    batch_size: int = 32,
    image_batch_size: int = 8,
    max_chunks: int | None = None,
    allow_model_download: bool = False,
) -> dict[str, Any]:
    """Create a new, complete vector database without changing parsed chunks."""
    if batch_size < 1 or image_batch_size < 1 or (max_chunks is not None and max_chunks < 1):
        raise ValueError("Batch sizes and max_chunks must be positive")
    chunks_path = chunks_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Vector DB already exists: {output_dir}. Choose a new output directory.")
    spaces = _load_records(chunks_path, max_chunks)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vector-db-build-", dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir) / "db"
        staging.mkdir()
        manifest: dict[str, Any] = {
            "format_version": 1,
            "source_chunks": str(chunks_path),
            "source_sha256": _sha256(chunks_path),
            "max_chunks": max_chunks,
            "spaces": {},
        }
        text_chunks = sum(len(spaces[name]) for name in TEXT_SPACES)
        if text_chunks:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(text_model, device=device, local_files_only=not allow_model_download)
            for name in TEXT_SPACES:
                if spaces[name]:
                    manifest["spaces"][name] = _write_text_space(staging, name, spaces[name], model, text_model, batch_size)
        if spaces[IMAGE_SPACE]:
            manifest["spaces"][IMAGE_SPACE] = _write_image_space(
                staging, spaces[IMAGE_SPACE], clip_model, clip_pretrained, device, image_batch_size,
            )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output_dir)
    return manifest
