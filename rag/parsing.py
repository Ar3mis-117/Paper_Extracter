"""PDF -> section-aware chunks with page numbers."""
import re
from dataclasses import dataclass
from pathlib import Path

SECTION_NAMES = {
    "abstract", "introduction", "background", "methods", "method", "materials and methods",
    "results", "discussion", "results and discussion", "conclusion", "conclusions",
    "references", "limitations",
}


@dataclass
class Chunk:
    id: str
    idx: int          # position within the paper (keeps reading order)
    paper_id: str
    section: str
    page: int         # 1-based
    text: str


def _page_number(meta: dict, fallback: int) -> int:
    """1-based page number across pymupdf4llm versions."""
    if "page_number" in meta:
        return int(meta["page_number"])
    if "page" in meta:
        return int(meta["page"]) + 1
    return fallback


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs up to `size` chars; hard-split oversized blocks (e.g. big tables) with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(p) > size:
            if cur:
                out.append(cur)
                cur = ""
            step = max(size - overlap, 1)
            out += [p[i:i + size] for i in range(0, len(p), step)]
        elif len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return [o for o in out if len(o) > 40]


def parse_pdf(path: Path, paper_id: str, chunk_chars: int = 1400, overlap: int = 200) -> list[Chunk]:
    import pymupdf4llm  # imported lazily so other modules work without it

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    pieces: list[tuple[str, int, str]] = []  # (section, page, text)
    section = "Front matter"
    for i, page in enumerate(pages, start=1):
        pno = _page_number(page["metadata"], i)
        buf = ""
        for line in page["text"].splitlines():
            m = re.match(r"^#{1,4}\s+(.*)", line)
            plain = re.sub(r"^[\d.\s]*|[*_:\s]*$", "", line).lower()  # "2. Methods", "**Results**"
            if m or plain in SECTION_NAMES:
                pieces.append((section, pno, buf))
                buf = ""
                section = (m.group(1) if m else line).strip(" *_#:")
            buf += line + "\n"
        pieces.append((section, pno, buf))

    chunks: list[Chunk] = []
    for sec, pno, text in pieces:
        for part in _split(text, chunk_chars, overlap):
            n = len(chunks)
            chunks.append(Chunk(id=f"{paper_id}::{n}", idx=n, paper_id=paper_id,
                                section=sec, page=pno, text=part))
    return chunks
