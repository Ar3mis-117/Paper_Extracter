"""Dense embeddings (sentence-transformers)."""
import numpy as np


class Embedder:
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() and "en" in model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(v, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode([self._prefix + text], normalize_embeddings=True)
        return np.asarray(v, dtype="float32")
