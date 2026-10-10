"""Runtime configuration for the online RAG workflow."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    chunks_path: Path
    vector_db_dir: Path
    keyword_db_path: Path
    access_token: str
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    rerank_model: str
    device: str
    tracing_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        # Existing reference-project variables are accepted by the current SDK.
        aliases = {
            "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
            "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
            "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
            "LANGSMITH_TRACING": "LANGCHAIN_TRACING_V2",
        }
        for current, legacy in aliases.items():
            if not os.getenv(current) and os.getenv(legacy):
                os.environ[current] = os.environ[legacy]
        provider = os.getenv("RAG_LLM_PROVIDER", "google").lower()
        default_model = "gemini-3.8-flash" if provider == "google" else os.getenv("OLLAMA_MODEL", "llama3.1")
        return cls(
            chunks_path=Path(os.getenv("RAG_CHUNKS_PATH", "data/processed/tata_motors_doc/chunks.jsonl")).resolve(),
            vector_db_dir=Path(os.getenv("RAG_VECTOR_DB_DIR", "data/vector_db/tata_motors_doc")).resolve(),
            keyword_db_path=Path(os.getenv("RAG_KEYWORD_DB_PATH", "data/search_index/tata_motors_doc.sqlite3")).resolve(),
            access_token=os.getenv("RAG_ACCESS_TOKEN", ""),
            llm_provider=provider,
            llm_model=os.getenv("RAG_LLM_MODEL", default_model),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            rerank_model=os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base"),
            device=os.getenv("RAG_DEVICE", "cpu"),
            tracing_enabled=os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"},
        )

    def authenticate(self, candidate: str) -> bool:
        return bool(self.access_token) and hmac.compare_digest(candidate, self.access_token)

    def validate_runtime(self) -> None:
        if not self.access_token:
            raise ValueError("Set RAG_ACCESS_TOKEN in .env before starting the frontend")
        if self.llm_provider not in {"google", "ollama"}:
            raise ValueError("RAG_LLM_PROVIDER must be google or ollama")
        if self.llm_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is required for the Google multimodal provider")
        for path, label in (
            (self.chunks_path, "Parsed chunks"),
            (self.vector_db_dir / "manifest.json", "Vector DB"),
            (self.keyword_db_path, "BM25 index"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing: {path}")
