"""
Offline end-to-end test: real PDF parsing, FAISS, BM25, fusion, extraction and tables,
with stand-ins for the embedding model, reranker and Gemini (no downloads, no API key).
Run:  python tests/smoke_test.py
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import PaperPipeline, Settings, build_tables  # noqa: E402


class FakeEmbedder:  # hashed bag-of-words
    def _vec(self, t):
        v = np.zeros(256, dtype="float32")
        for w in re.findall(r"\w+", t.lower()):
            v[hash(w) % 256] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, t):
        return self._vec(t)[None, :]


class FakeReranker:  # token-overlap score
    def rerank(self, query, chunks, top_k):
        q = set(re.findall(r"\w+", query.lower()))
        return sorted(chunks, key=lambda c: -len(q & set(re.findall(r"\w+", c.text.lower()))))[:top_k]


class FakeLLM:  # "finds" the sentence mentioning participants; quotes it verbatim
    def generate(self, system, prompt, schema=None):
        if schema is None:
            return "answer (paper, p.1)"
        m = re.search(r"[^.\n]*participants[^.\n]*", prompt)
        quote = m.group(0).strip() if m else ""
        item = {"value": "120" if quote else "not reported", "evidence": quote, "page": 99 if quote else None}
        return json.dumps({k: item for k in schema.model_fields})


def make_pdf(path, n):
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), f"Abstract\nThis is paper number {n} about a clinical trial of a new treatment.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), f"Methods\nWe enrolled {100 + n * 20} participants across three hospitals in 2021.", fontsize=11)
    doc.save(path)


with tempfile.TemporaryDirectory() as tmp:
    pdfs = []
    for n in (1, 2):
        f = Path(tmp) / f"paper{n}.pdf"
        make_pdf(f, n)
        pdfs.append(f)

    cfg = Settings(data_dir=str(Path(tmp) / "data"), candidate_k=6, top_k_per_field=3)
    pipe = PaperPipeline(cfg, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())
    names = {pipe.add_pdf(f.name, f.read_bytes()): f.stem for f in pdfs}
    assert len(pipe.store.paper_ids()) == 2

    # a fresh pipeline must find the persisted FAISS indexes
    pipe2 = PaperPipeline(cfg, embedder=FakeEmbedder(), reranker=None, llm=FakeLLM())
    assert pipe2.store.paper_ids() == pipe.store.paper_ids()

    for p in (pipe, pipe2):
        results, errors = p.extract(list(names), {"Sample size": "number of participants"})
        assert not errors, errors
        wide, long = build_tables(results, names, {"sample_size": ""})
        print(wide.to_string(index=False)); print(long[["Paper", "Page", "Section", "Quote verified"]].to_string(index=False))
        assert (long["Page"] == 2).all(), "page must come from the verified chunk (2), not the model's guess (99)"
        assert long["Quote verified"].all()
        assert (long["Section"] == "Methods").all()
    print("ask ->", pipe.ask("how many participants?"))
    print("\nOK: smoke test passed")
