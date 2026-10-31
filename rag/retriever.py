"""Hybrid retrieval: FAISS (dense) + BM25 (sparse) -> reciprocal rank fusion -> optional reranker."""
import threading

from .config import Settings
from .keyword_search import BM25Index
from .parsing import Chunk
from .vector_store import IndexStore


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several best-first rankings."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, store: IndexStore, embedder, reranker, settings: Settings):
        self.store, self.embedder, self.reranker, self.s = store, embedder, reranker, settings
        self._bm25: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def _bm25_for(self, paper_id: str) -> BM25Index:
        with self._lock:
            if paper_id not in self._bm25:
                self._bm25[paper_id] = BM25Index([c.text for c in self.store.get(paper_id).chunks])
            return self._bm25[paper_id]

    def candidates(self, query: str, paper_id: str, n: int) -> list[Chunk]:
        """Top-n fused candidates (before reranking) from one paper."""
        index = self.store.get(paper_id)
        dense = [i for i, _ in index.search(self.embedder.embed_query(query), self.s.dense_k)]
        sparse = [i for i, _ in self._bm25_for(paper_id).search(query, self.s.bm25_k)]
        return [index.chunks[i] for i in rrf([dense, sparse])[:n]]

    def search(self, query: str, paper_id: str, k: int | None = None) -> list[Chunk]:
        k = k or self.s.top_k_per_field
        if self.reranker is None:
            return self.candidates(query, paper_id, k)
        cands = self.candidates(query, paper_id, max(self.s.candidate_k, k))
        return self.reranker.rerank(query, cands, k)

    def search_many(self, query: str, paper_ids: list[str], k: int = 8) -> list[Chunk]:
        """Search across several papers and return the best k overall."""
        per_paper = [self.candidates(query, pid, max(self.s.candidate_k, k)) for pid in paper_ids]
        if self.reranker is not None:
            return self.reranker.rerank(query, [c for lst in per_paper for c in lst], k)
        merged = []  # round-robin so every paper is represented
        for rank in range(max((len(l) for l in per_paper), default=0)):
            merged += [l[rank] for l in per_paper if rank < len(l)]
        return merged[:k]
