"""Build the BM25 index used by the online graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_rag.keyword_index import build_keyword_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite FTS5/BM25 search index")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/tata_motors_doc/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/search_index/tata_motors_doc.sqlite3"))
    args = parser.parse_args()
    count = build_keyword_index(args.chunks, args.output)
    print(f"Indexed {count} chunks in {args.output.resolve()}")


if __name__ == "__main__":
    main()
