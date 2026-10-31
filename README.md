# Paper data extractor (FAISS + BM25 + reranker + Gemini)

Upload several scientific papers, name the fields you want, get one table where every value
carries a page reference and a verifiable evidence quote.

```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
streamlit run app.py                 # web app
python cli.py ingest ./papers        # or CLI
python cli.py extract --fields-file fields.json
python tests/smoke_test.py           # offline self-test, no key or downloads needed
```

## Layout (one file per component)

| File | Role |
|---|---|
| `rag/config.py` | every tunable setting (models, chunk size, retrieval depths, workers) |
| `rag/parsing.py` | PDF -> section-aware chunks with page numbers (pymupdf4llm) |
| `rag/embeddings.py` | dense embeddings (sentence-transformers, bge) |
| `rag/vector_store.py` | FAISS, one index per paper, saved to `data/index/` |
| `rag/keyword_search.py` | BM25 keyword search |
| `rag/retriever.py` | FAISS + BM25 -> reciprocal rank fusion -> reranker |
| `rag/reranker.py` | cross-encoder reranker (bge-reranker) |
| `rag/llm.py` | Gemini client with retry/backoff on rate limits |
| `rag/schemas.py` | extraction schema built from the user's fields |
| `rag/extractor.py` | per-paper extraction + quote verification (page/section taken from the matched chunk) |
| `rag/tables.py` | results -> table, references table, Excel export |
| `rag/pipeline.py` | orchestration: ingest, parallel extraction, Q&A |
| `app.py` / `cli.py` | Streamlit UI / command line |

Each component can be swapped independently: for example replace `llm.py` with another provider
exposing `generate(system, prompt, schema=None)`, or pass `reranker=None` to disable reranking.
