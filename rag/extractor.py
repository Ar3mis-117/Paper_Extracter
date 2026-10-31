"""Per-paper structured extraction with evidence verification."""
import re
import unicodedata

from pydantic import BaseModel

from .config import Settings
from .retriever import HybridRetriever

SYSTEM = (
    "You extract data from scientific papers. Use ONLY the provided excerpts. "
    "If a field is not stated, set value to 'not reported', evidence to '' and page to null. "
    "Copy numbers and units exactly. Evidence must be a short verbatim quote from the excerpts."
)


def _norm(s: str) -> str:
    """Normalise text so quote matching ignores case, markdown marks, dashes and whitespace."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[*_#`]", "", s)
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def format_context(chunks) -> str:
    return "\n\n".join(f"[p.{c.page} | {c.section}]\n{c.text}" for c in chunks)


class PaperExtractor:
    def __init__(self, retriever: HybridRetriever, llm, settings: Settings):
        self.retriever, self.llm, self.s = retriever, llm, settings

    def extract(self, paper_id: str, model: type[BaseModel], fields: dict[str, str]) -> dict:
        # 1. retrieve per field (name + hint is the query), union the results in reading order
        seen, ctx = set(), []
        for name, hint in fields.items():
            for c in self.retriever.search(f"{name.replace('_', ' ')}: {hint}", paper_id):
                if c.id not in seen:
                    seen.add(c.id)
                    ctx.append(c)
        ctx.sort(key=lambda c: c.idx)

        # 2. ask Gemini for schema-constrained JSON
        text = self.llm.generate(SYSTEM, f"Paper: {paper_id}\n\n{format_context(ctx)}", schema=model)
        result = model.model_validate_json(text).model_dump()

        # 3. verify every quote against the source; on a match, take page/section from the chunk itself
        joined = _norm(" ".join(c.text for c in ctx))
        for item in result.values():
            q = _norm(item["evidence"])
            item["quote_verified"], item["section"] = False, ""
            if not q:
                continue
            hit = next((c for c in ctx if q in _norm(c.text)), None)
            if hit:
                item.update(quote_verified=True, page=hit.page, section=hit.section)
            elif q in joined:  # quote spans two adjacent chunks
                item["quote_verified"] = True
        return result
