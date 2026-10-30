from paper_rag.models import ExtractedValue, FieldSpec, PaperResult, Reference, results_to_rows


def test_rows_have_values_and_references():
    fields = [FieldSpec("Sample size"), FieldSpec("Method")]
    res = PaperResult("a.pdf", {
        "Sample size": ExtractedValue("Sample size", "120", [Reference("a.pdf", 3, "n = 120")]),
    })
    row = results_to_rows([res], fields)[0]
    assert row["Sample size"] == "120"
    assert row["Method"] == "Not found"
    assert row["References"] == "Sample size: p.3"
