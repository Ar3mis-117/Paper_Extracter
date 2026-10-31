"""Sparse (BM25) keyword search - catches exact terms like gene names, compounds and acronyms."""
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
