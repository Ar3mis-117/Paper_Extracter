"""Extraction schema: built at run time from the fields the user asks for."""
import re

from pydantic import BaseModel, Field, create_model


class Item(BaseModel):
    value: str = Field(description="Extracted value, or 'not reported' if absent. Never guess.")
    evidence: str = Field(description="Short verbatim quote supporting the value; empty string if not reported.")
    page: int | None = Field(description="Page number from the [p.N] tag of the evidence; null if not reported.")


def normalize_fields(fields: dict[str, str]) -> dict[str, str]:
    """Turn user labels into safe identifiers; the original label becomes the hint if none is given."""
    clean = {}
    for name, desc in fields.items():
        key = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")
        if key:
            clean[key] = (desc or "").strip() or str(name).strip()
    return clean


def build_model(fields: dict[str, str]) -> type[BaseModel]:
    """Create the Pydantic schema (one Item per field) used for Gemini's structured output."""
    return create_model(
        "PaperExtraction",
        **{name: (Item, Field(description=desc)) for name, desc in fields.items()},
    )
