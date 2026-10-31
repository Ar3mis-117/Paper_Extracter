"""All tunable settings in one place."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    data_dir: str = "data"

    # models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    llm_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    use_reranker: bool = True

    # chunking
    chunk_chars: int = 1400
    chunk_overlap: int = 200

    # retrieval
    dense_k: int = 20          # candidates from FAISS
    bm25_k: int = 20           # candidates from BM25
    candidate_k: int = 12      # fused candidates handed to the reranker
    top_k_per_field: int = 4   # chunks finally kept per requested field

    # extraction
    max_workers: int = 2       # papers processed in parallel (keep low for Gemini free-tier limits)

    @property
    def index_dir(self) -> Path:
        return Path(self.data_dir) / "index"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"
