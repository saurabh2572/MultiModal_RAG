"""Query one vector space from a built modality-based FAISS database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from multimodal_rag.vector_db import IMAGE_SPACE, TEXT_SPACES


def _query_vector(space: str, description: dict, query: str | None, image_query: Path | None, device: str) -> np.ndarray:
    if space in TEXT_SPACES:
        if not query or image_query:
            raise ValueError("Text and table spaces require a text query")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(description["model"], device=device, local_files_only=True)
        vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return np.ascontiguousarray(vector, dtype="float32")

    if space != IMAGE_SPACE:
        raise ValueError(f"Unknown space: {space}")
    if bool(query) == bool(image_query):
        raise ValueError("Image visual search needs exactly one of --query or --image-query")
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        description["model"], pretrained=description["pretrained"], device=device,
    )
    model.eval()
    with torch.inference_mode():
        if image_query:
            with Image.open(image_query) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            features = model.encode_image(tensor)
        else:
            tokenizer = open_clip.get_tokenizer(description["model"])
            features = model.encode_text(tokenizer([query]).to(device))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return np.ascontiguousarray(features.float().cpu().numpy(), dtype="float32")


def search_vector_db(
    db_dir: Path,
    space: str,
    *,
    query: str | None = None,
    image_query: Path | None = None,
    top_k: int = 5,
    device: str = "cpu",
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    db_dir = db_dir.resolve(strict=True)
    manifest = json.loads((db_dir / "manifest.json").read_text(encoding="utf-8"))
    description = manifest["spaces"].get(space)
    if not description:
        raise ValueError(f"Space {space!r} is not in this database; available: {list(manifest['spaces'])}")
    space_dir = db_dir / space
    index = faiss.read_index(str(space_dir / "index.faiss"))
    records = [json.loads(line) for line in (space_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if index.ntotal != description["count"] or len(records) != index.ntotal:
        raise ValueError(f"Index and record counts differ for {space}")
    vector = _query_vector(space, description, query, image_query, device)
    if vector.shape[1] != index.d:
        raise ValueError(f"Query dimension {vector.shape[1]} does not match index dimension {index.d}")
    faiss.normalize_L2(vector)
    scores, positions = index.search(vector, min(top_k, index.ntotal))
    return [
        {"score": float(score), **records[int(position)]}
        for score, position in zip(scores[0], positions[0]) if position >= 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search one modality-based vector index")
    parser.add_argument("--db-dir", type=Path, default=Path("data/vector_db/tata_motors_doc"))
    parser.add_argument("--space", choices=(*TEXT_SPACES, IMAGE_SPACE), required=True)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument("--image-query", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = search_vector_db(
        args.db_dir, args.space, query=args.query, image_query=args.image_query,
        top_k=args.top_k, device=args.device,
    )
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
