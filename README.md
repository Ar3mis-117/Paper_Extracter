# RAG

RAG pipeline that extracts **user-chosen fields** from **multiple scientific papers**
and returns a table with **references** (paper + page) for every value.

## Roadmap
1. [x] Scaffold + data models
2. [ ] PDF loading -> page-aware chunks
3. [ ] Embeddings + FAISS index (per paper)
4. [ ] Per-field retrieval
5. [ ] LLM extraction (Gemini) with structured JSON + quotes
6. [ ] Streamlit UI: upload PDFs, enter fields, view/download table

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your key
pytest
```
