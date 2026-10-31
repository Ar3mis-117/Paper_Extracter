"""
Streamlit app: upload papers -> choose fields -> table with page references and evidence quotes.
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from rag import PaperPipeline, Settings, build_tables, normalize_fields, to_excel_bytes

st.set_page_config(page_title="Paper data extractor", layout="wide")
st.title("Scientific paper data extractor")
st.caption("Upload papers, choose what to extract, get a table where every value points back to its source.")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(name: str):
    from rag.embeddings import Embedder
    return Embedder(name)


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker(name: str):
    from rag.reranker import Reranker
    return Reranker(name)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", type="password",
                            help="Leave empty to use the GEMINI_API_KEY environment variable.")
    base = Settings()
    llm_model = st.text_input("Gemini model", value=base.llm_model)
    use_reranker = st.checkbox("Use reranker (slower, more accurate)", value=True)
    top_k = st.slider("Chunks retrieved per field", 2, 10, base.top_k_per_field)
    workers = st.slider("Papers processed in parallel", 1, 6, base.max_workers,
                        help="Lower this if you hit Gemini rate limits.")
    st.markdown("**How references work**\n\nEach cell shows the value and its page. The *References* tab "
                "lists the exact quote and section. `(unverified)` means the quote was not found verbatim "
                "in the retrieved text, so check those by hand.")

# ------------------------------------------------------------------- inputs
files = st.file_uploader("1. Upload scientific papers (PDF)", type="pdf", accept_multiple_files=True)

st.subheader("2. Fields to extract")
default_rows = pd.DataFrame([
    {"Field": "Study design", "Hint (optional)": "e.g. RCT, cohort, simulation"},
    {"Field": "Sample size", "Hint (optional)": "Number of participants or samples"},
    {"Field": "Primary outcome", "Hint (optional)": ""},
    {"Field": "Key results", "Hint (optional)": "Main quantitative findings with effect sizes, p-values"},
])
edited = st.data_editor(default_rows, num_rows="dynamic", use_container_width=True, key="fields")
run = st.button("Extract data", type="primary", disabled=not files)

# --------------------------------------------------------------- extraction
if run:
    raw = {str(r["Field"]).strip(): str(r["Hint (optional)"] or "").strip()
           for _, r in edited.iterrows() if str(r["Field"]).strip() not in ("", "None", "nan")}
    fields = normalize_fields(raw)
    if not fields:
        st.error("Add at least one field.")
        st.stop()

    settings = Settings(llm_model=llm_model, use_reranker=use_reranker,
                        top_k_per_field=top_k, max_workers=workers)
    try:
        pipe = PaperPipeline(
            settings, api_key=api_key or None,
            embedder=load_embedder(settings.embed_model),
            reranker=load_reranker(settings.rerank_model) if use_reranker else None,
        )
        _ = pipe.llm  # fail early if the API key is missing
    except Exception as e:
        st.error(str(e))
        st.stop()

    names, errors = {}, {}
    with st.spinner("Reading and indexing papers..."):
        for f in files:
            try:
                names[pipe.add_pdf(f.name, f.getvalue())] = f.name.rsplit(".", 1)[0]
            except Exception as e:
                errors[f.name] = str(e)

    bar = st.progress(0.0, text="Extracting...")
    results, ex_errors = pipe.extract(
        list(names), fields,
        on_progress=lambda done, total, pid: bar.progress(done / total, text=f"Extracted {done}/{total} papers"),
    )
    errors.update({names[p]: m for p, m in ex_errors.items()})
    bar.progress(1.0, text="Done")

    wide, long = build_tables(results, names, fields)
    st.session_state["out"] = {"wide": wide, "long": long, "errors": errors}

# ------------------------------------------------------------------ results
out = st.session_state.get("out")
if out:
    for paper, msg in out["errors"].items():
        st.warning(f"{paper}: {msg}")
    if out["wide"].empty:
        st.stop()

    st.subheader("3. Results")
    tab_table, tab_refs = st.tabs(["Table", "References"])
    with tab_table:
        st.dataframe(out["wide"], use_container_width=True, hide_index=True)
    with tab_refs:
        st.dataframe(out["long"], use_container_width=True, hide_index=True,
                     column_config={"Evidence (quote)": st.column_config.TextColumn(width="large")})
        long = out["long"]
        unverified = (~long["Quote verified"] & (long["Value"].str.strip().str.lower() != "not reported")).sum()
        if unverified:
            st.info(f"{unverified} value(s) have quotes that could not be matched verbatim. Please check them manually.")

    c1, c2 = st.columns(2)
    c1.download_button("Download CSV (with references)", out["long"].to_csv(index=False).encode(),
                       "extraction_references.csv", "text/csv")
    c2.download_button("Download Excel (table + references)", to_excel_bytes(out["wide"], out["long"]),
                       "extraction.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
