"""Cross-encoder reranker: re-scores (query, chunk) pairs jointly for much better precision."""
from .parsing import Chunk


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c.text) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))[:top_k]
        return [chunks[i] for i in order]
