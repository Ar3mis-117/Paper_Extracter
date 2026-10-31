"""Orchestration: ingest PDFs -> index -> extract fields from many papers -> results."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Settings
from .extractor import PaperExtractor, format_context
from .parsing import parse_pdf
from .retriever import HybridRetriever
from .schemas import build_model, normalize_fields
from .vector_store import IndexStore


class PaperPipeline:
    """
    Components are created lazily, or injected (handy for caching in Streamlit and for tests):
    embedder, reranker (None disables reranking) and llm.
    """

    _UNSET = object()

    def __init__(self, settings: Settings | None = None, api_key: str | None = None,
                 embedder=None, reranker=_UNSET, llm=None):
        self.s = settings or Settings()
        self.api_key = api_key
        self._embedder, self._reranker, self._llm = embedder, reranker, llm
        self.s.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.store = IndexStore(self.s.index_dir)
        self._retriever = None

    # ---- lazily built components
    @property
    def embedder(self):
        if self._embedder is None:
            from .embeddings import Embedder
            self._embedder = Embedder(self.s.embed_model)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is self._UNSET:
            if self.s.use_reranker:
                from .reranker import Reranker
                self._reranker = Reranker(self.s.rerank_model)
            else:
                self._reranker = None
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from .llm import GeminiClient
            self._llm = GeminiClient(self.api_key, self.s.llm_model)
        return self._llm

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.store, self.embedder, self.reranker, self.s)
        return self._retriever

    # ---- ingestion
    def add_pdf(self, name: str, data: bytes) -> str:
        """Save + index one PDF (skipped if this exact file is already indexed). Returns its paper_id."""
        stem = re.sub(r"[^\w\-]+", "_", Path(name).stem)[:60]
        paper_id = f"{stem}_{hashlib.md5(data).hexdigest()[:6]}"  # content hash: same name, different file -> different id
        path = self.s.uploads_dir / f"{paper_id}.pdf"
        path.write_bytes(data)
        if self.store.has(paper_id):
            return paper_id
        chunks = parse_pdf(path, paper_id, self.s.chunk_chars, self.s.chunk_overlap)
        if not chunks:
            raise ValueError("No readable text found (scanned PDF? run OCR first).")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(paper_id, chunks, vectors)
        return paper_id

    def add_directory(self, folder: str) -> dict[str, str]:
        """Index every PDF in a folder. Returns {paper_id: display name}."""
        out = {}
        for pdf in sorted(Path(folder).glob("*.pdf")):
            out[self.add_pdf(pdf.name, pdf.read_bytes())] = pdf.stem
        return out

    # ---- extraction
    def extract(self, paper_ids: list[str], fields: dict[str, str], on_progress=None):
        """
        Extract `fields` ({name: hint}) from each paper. Returns (results, errors), both keyed by paper_id.
        on_progress(done, total, paper_id) is called from the calling thread as papers finish.
        """
        fields = normalize_fields(fields)
        model = build_model(fields)
        extractor = PaperExtractor(self.retriever, self.llm, self.s)
        _ = self.retriever, self.llm  # build shared components before threads start
        results, errors = {}, {}
        with ThreadPoolExecutor(max_workers=self.s.max_workers) as ex:
            futures = {ex.submit(extractor.extract, pid, model, fields): pid for pid in paper_ids}
            for done, fut in enumerate(as_completed(futures), start=1):
                pid = futures[fut]
                try:
                    results[pid] = fut.result()
                except Exception as e:  # one failing paper must not lose the others
                    errors[pid] = str(e)
                if on_progress:
                    on_progress(done, len(paper_ids), pid)
        return {p: results[p] for p in paper_ids if p in results}, errors

    # ---- free-form Q&A
    def ask(self, question: str, paper_ids: list[str] | None = None) -> str:
        ids = paper_ids or self.store.paper_ids()
        ctx = self.retriever.search_many(question, ids, k=8)
        return self.llm.generate(
            "Answer using only the excerpts. Cite as (paper, p.X). If the answer isn't in the excerpts, say so.",
            f"Question: {question}\n\n" + "\n\n".join(
                f"[{c.paper_id} p.{c.page} | {c.section}]\n{c.text}" for c in ctx),
        )
