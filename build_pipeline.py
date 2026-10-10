"""Run PDF ingestion, then build independent retrieval indexes concurrently."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


async def _run(*command: str) -> None:
    process = await asyncio.create_subprocess_exec(sys.executable, *command)
    try:
        return_code = await process.wait()
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        raise
    if return_code:
        raise RuntimeError(f"{Path(command[0]).name} failed with exit code {return_code}")


async def build_pipeline(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    if args.vector_db_dir.exists() or args.keyword_db_path.exists():
        raise FileExistsError("Choose unused vector and BM25 output paths before starting ingestion")
    await _run(
        str(root / "main.py"), str(args.pdf), "--output-dir", str(args.output_dir),
        "--start-page", str(args.start_page), "--ocr", args.ocr,
        "--dpi", str(args.dpi), "--max-tokens", str(args.max_tokens),
        "--overlap-tokens", str(args.overlap_tokens),
        *(["--max-pages", str(args.max_pages)] if args.max_pages else []),
    )
    chunks = args.output_dir / "chunks.jsonl"
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(_run(
            str(root / "build_vector_db.py"), "--chunks", str(chunks),
            "--output-dir", str(args.vector_db_dir), "--device", args.device,
            *(["--allow-model-download"] if args.allow_model_download else []),
        ))
        tasks.create_task(_run(
            str(root / "build_search_index.py"), "--chunks", str(chunks),
            "--output", str(args.keyword_db_path),
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF and build vector and BM25 indexes")
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("data/raw/tata_motors_doc.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tata_motors_doc"))
    parser.add_argument("--vector-db-dir", type=Path, default=Path("data/vector_db/tata_motors_doc"))
    parser.add_argument("--keyword-db-path", type=Path, default=Path("data/search_index/tata_motors_doc.sqlite3"))
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--ocr", choices=("auto", "all", "off"), default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    asyncio.run(build_pipeline(args))


if __name__ == "__main__":
    main()
