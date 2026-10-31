"""
Command line:
  python cli.py ingest ./papers
  python cli.py extract [--fields-file fields.json] [--no-rerank]     -> results.csv / results.xlsx
  python cli.py ask "What dose was used?"
Needs GEMINI_API_KEY in the environment.
"""
import argparse
import json
import sys
from pathlib import Path

from rag import PaperPipeline, Settings, build_tables, normalize_fields, to_excel_bytes


def prompt_for_fields() -> dict[str, str]:
    print("What data should be extracted from each paper? Leave the name empty when done.\n")
    fields = {}
    while True:
        name = input("Field name (e.g. sample size): ").strip()
        if not name:
            return fields
        fields[name] = input("  Short hint (optional): ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-rerank", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("pdf_dir")
    ex = sub.add_parser("extract")
    ex.add_argument("--fields-file", help='JSON: {"field name": "hint"}')
    q = sub.add_parser("ask")
    q.add_argument("question")
    args = ap.parse_args()

    pipe = PaperPipeline(Settings(data_dir=args.data_dir, use_reranker=not args.no_rerank))

    if args.cmd == "ingest":
        papers = pipe.add_directory(args.pdf_dir)
        print(f"Indexed {len(papers)} papers: {', '.join(papers.values())}")
    elif args.cmd == "ask":
        print(pipe.ask(args.question))
    else:
        raw = json.loads(Path(args.fields_file).read_text()) if args.fields_file else prompt_for_fields()
        fields = normalize_fields(raw)
        if not fields:
            sys.exit("No fields given.")
        ids = pipe.store.paper_ids()
        results, errors = pipe.extract(ids, fields, on_progress=lambda d, t, p: print(f"[{d}/{t}] {p}"))
        wide, long = build_tables(results, {p: p for p in ids}, fields)
        long.to_csv("results.csv", index=False)
        Path("results.xlsx").write_bytes(to_excel_bytes(wide, long))
        for pid, msg in errors.items():
            print(f"FAILED {pid}: {msg}")
        print("Saved results.csv and results.xlsx")


if __name__ == "__main__":
    main()
