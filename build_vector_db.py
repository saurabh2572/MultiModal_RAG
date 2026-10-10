"""Command line entry point for modality-based vector indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_rag.vector_db import build_vector_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed parsed PDF chunks into separate FAISS vector spaces")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/tata_motors_doc/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vector_db/tata_motors_doc"))
    parser.add_argument("--text-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, help="Index the first N chunks for a trial run")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow downloading the BGE text model if absent")
    args = parser.parse_args()
    manifest = build_vector_db(
        args.chunks,
        args.output_dir,
        text_model=args.text_model,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
        image_batch_size=args.image_batch_size,
        max_chunks=args.max_chunks,
        allow_model_download=args.allow_model_download,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
