"""Turn extraction results into a table (with page references) plus a long 'references' table."""
import io

import pandas as pd


def _title(field: str) -> str:
    return field.replace("_", " ").title()


def build_tables(results: dict[str, dict], names: dict[str, str], fields: dict[str, str]):
    """
    Returns (wide, long) DataFrames.
      wide: one row per paper, one column per field; cell = "value [p.N]" (+ "(unverified)" if needed)
      long: one row per (paper, field) with section, evidence quote and verification flag
    """
    wide, long = [], []
    for pid, r in results.items():
        paper = names.get(pid, pid)
        row = {"Paper": paper}
        for f in fields:
            it = r[f]
            reported = it["value"].strip().lower() != "not reported"
            ref = f" [p.{it['page']}]" if it.get("page") else ""
            flag = " (unverified)" if reported and not it["quote_verified"] else ""
            row[_title(f)] = f"{it['value']}{ref}{flag}"
            long.append({
                "Paper": paper, "Field": _title(f), "Value": it["value"], "Page": it["page"],
                "Section": it.get("section", ""), "Evidence (quote)": it["evidence"],
                "Quote verified": it["quote_verified"],
            })
        wide.append(row)
    return pd.DataFrame(wide), pd.DataFrame(long)


def to_excel_bytes(wide: pd.DataFrame, long: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        wide.to_excel(xw, sheet_name="Table", index=False)
        long.to_excel(xw, sheet_name="References", index=False)
    return buf.getvalue()
