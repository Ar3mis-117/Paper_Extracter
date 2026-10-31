"""FAISS vector store: one index per paper (so 'search only this paper' is exact and cheap)."""
import json
import shutil
import threading
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .parsing import Chunk


class PaperIndex:
    def __init__(self, chunks: list[Chunk], index: faiss.Index):
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "PaperIndex":
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine on normalised vectors
        index.add(vectors)
        return cls(chunks, index)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "faiss.index"))
        (folder / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]))

    @classmethod
    def load(cls, folder: Path) -> "PaperIndex":
        index = faiss.read_index(str(folder / "faiss.index"))
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text())]
        return cls(chunks, index)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk position, score)] best first."""
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(query_vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


class IndexStore:
    """Manages the per-paper indexes on disk, with an in-memory cache."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PaperIndex] = {}
        self._lock = threading.Lock()

    def has(self, paper_id: str) -> bool:
        return (self.root / paper_id / "faiss.index").exists()

    def add(self, paper_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        idx = PaperIndex.build(chunks, vectors)
        idx.save(self.root / paper_id)
        with self._lock:
            self._cache[paper_id] = idx

    def get(self, paper_id: str) -> PaperIndex:
        with self._lock:
            if paper_id not in self._cache:
                self._cache[paper_id] = PaperIndex.load(self.root / paper_id)
            return self._cache[paper_id]

    def paper_ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "faiss.index").exists())

    def delete(self, paper_id: str) -> None:
        with self._lock:
            self._cache.pop(paper_id, None)
        shutil.rmtree(self.root / paper_id, ignore_errors=True)
