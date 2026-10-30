"""Core data models shared by every stage of the pipeline.

Flow: PDF -> Chunk(s) -> retrieval -> ExtractedValue(+References) -> table rows
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    """A field the END USER asks us to extract (no fixed schema)."""
    name: str                   # column header, e.g. "Sample size"
    description: str = ""       # helps the LLM, e.g. "number of participants"


@dataclass(frozen=True)
class Chunk:
    paper_id: str               # usually the PDF filename
    page: int                   # 1-based
    text: str
    chunk_id: int = 0


@dataclass(frozen=True)
class Reference:
    paper_id: str
    page: int
    quote: str                  # supporting snippet from the paper


@dataclass
class ExtractedValue:
    field: str
    value: str | None           # None when not found in the paper
    references: list[Reference] = field(default_factory=list)


@dataclass
class PaperResult:
    paper_id: str
    values: dict[str, ExtractedValue] = field(default_factory=dict)


def results_to_rows(results: list[PaperResult], fields: list[FieldSpec]) -> list[dict]:
    """Flatten results into table rows: one row per paper, one column per field,
    plus a 'References' column listing 'page N' citations."""
    rows = []
    for r in results:
        row: dict = {"Paper": r.paper_id}
        refs: list[str] = []
        for f in fields:
            ev = r.values.get(f.name)
            row[f.name] = ev.value if ev and ev.value is not None else "Not found"
            if ev:
                refs += [f"{f.name}: p.{x.page}" for x in ev.references]
        row["References"] = "; ".join(refs)
        rows.append(row)
    return rows
