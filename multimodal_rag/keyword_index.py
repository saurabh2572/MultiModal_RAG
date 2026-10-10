"""Persistent SQLite FTS5/BM25 index for exact-term retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import threading
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_keyword_index(chunks_path: Path, output_path: Path) -> int:
    chunks_path = chunks_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"BM25 index exists: {output_path}. Choose a new output path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bm25-build-", dir=output_path.parent) as directory:
        staged = Path(directory) / "index.sqlite3"
        connection = sqlite3.connect(staged)
        try:
            connection.executescript("""
                CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE VIRTUAL TABLE documents USING fts5(
                    chunk_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2'
                );
            """)
            count = 0
            with connection, chunks_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if not chunk.get("id") or not chunk.get("text", "").strip():
                        raise ValueError(f"Invalid chunk at line {line_number}")
                    connection.execute("INSERT INTO documents (chunk_id, text) VALUES (?, ?)", (chunk["id"], chunk["text"]))
                    count += 1
                if not count:
                    raise ValueError("No chunks found")
                connection.execute("INSERT INTO manifest VALUES (?, ?)", ("source_sha256", file_sha256(chunks_path)))
                connection.execute("INSERT INTO manifest VALUES (?, ?)", ("chunk_count", str(count)))
        finally:
            connection.close()
        staged.rename(output_path)
    return count


class KeywordIndex:
    def __init__(self, path: Path, *, expected_sha256: str):
        path = path.resolve(strict=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        stored = self.connection.execute("SELECT value FROM manifest WHERE key='source_sha256'").fetchone()
        if stored is None or stored["value"] != expected_sha256:
            self.connection.close()
            raise ValueError("BM25 index was built from different chunks.jsonl")

    def search(self, query: str, limit: int = 20) -> list[dict]:
        with self._lock:
            return self._search(query, limit)

    def _search(self, query: str, limit: int) -> list[dict]:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.connection.execute(
            "SELECT chunk_id, bm25(documents) AS score FROM documents "
            "WHERE documents MATCH ? ORDER BY score LIMIT ?",
            (match_query, limit),
        ).fetchall()
        return [{"chunk_id": row["chunk_id"], "score": float(row["score"]), "rank": index}
                for index, row in enumerate(rows, 1)]

    def close(self) -> None:
        with self._lock:
            self.connection.close()
