"""Central settings, overridable via environment variables."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    chunk_size: int = 1000      # characters per chunk (tuned in a later step)
    chunk_overlap: int = 150
    top_k: int = 5              # chunks retrieved per (paper, field)


settings = Settings()
