"""CLI for extracting and chunking a PDF into JSONL retrieval records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from multimodal_rag.chunker import chunk_elements
from multimodal_rag.parser import parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF and build modality aware chunks")
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("data/raw/tata_motors_doc.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tata_motors_doc"))
    parser.add_argument("--max-pages", type=int, default=None, help="Process the first N pages for a trial run")
    parser.add_argument("--start-page", type=int, default=1, help="First PDF page, counting from 1")
    parser.add_argument("--ocr", choices=("auto", "all", "off"), default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be positive")
    if args.start_page < 1:
        parser.error("--start-page must be positive")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    elements_path = args.output_dir / "elements.jsonl"
    chunks_path = args.output_dir / "chunks.jsonl"
    counts: Counter[str] = Counter()
    chunk_count = 0
    page_elements: list[dict] = []

    def write_chunks(chunk_file) -> None:
        nonlocal chunk_count
        for chunk in chunk_elements(page_elements, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens):
            chunk_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1
        page_elements.clear()

    with elements_path.open("w", encoding="utf-8") as element_file, chunks_path.open("w", encoding="utf-8") as chunk_file:
        for element in parse_pdf(args.pdf, args.output_dir / "images", start_page=args.start_page, max_pages=args.max_pages, ocr_mode=args.ocr, dpi=args.dpi):
            if page_elements and element["page"] != page_elements[-1]["page"]:
                write_chunks(chunk_file)
            element_file.write(json.dumps(element, ensure_ascii=False) + "\n")
            counts[element["modality"]] += 1
            page_elements.append(element)
        if page_elements:
            write_chunks(chunk_file)
    manifest = {
        "source": str(args.pdf.resolve()), "elements": dict(counts), "chunks": chunk_count,
        "ocr_mode": args.ocr, "start_page": args.start_page, "max_pages": args.max_pages,
        "elements_file": str(elements_path.resolve()), "chunks_file": str(chunks_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
