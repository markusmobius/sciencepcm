"""Print run files side by side, with titles, for human judgement.

Every configuration we have tried loses to plain BM25 on BioASQ nDCG@10, while
reranking consistently improves recall at depth. That combination is what a shallow
gold standard produces: only ~10% of BioASQ's judgements exist in this corpus, so a
system that surfaces an excellent unjudged paper is scored as if it had failed.

Metrics cannot settle that. Reading the results can. Gold documents are marked so the
judged and unjudged hits are distinguishable at a glance.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

import duckdb


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, help="Repeat per system.")
    parser.add_argument("--label", action="append", default=None, help="Optional name per --run.")
    parser.add_argument("--qrels", required=True, type=Path, help="Supplies query_text; gold is optional.")
    parser.add_argument("--articles", required=True, help="Glob for the abstracts Parquet")
    parser.add_argument("--queries", type=int, default=5, help="How many queries to show.")
    parser.add_argument("--top", type=int, default=10, help="Hits per system.")
    parser.add_argument("--query-id", action="append", default=None, help="Show these specific queries.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--width", type=int, default=110)
    args = parser.parse_args()

    labels = args.label or [path.stem for path in args.run]
    if len(labels) != len(args.run):
        parser.error("--label must be given as often as --run")

    runs = [{str(r["query_id"]): r["hits"] for r in load_jsonl(path)} for path in args.run]
    qrels = {str(r["query_id"]): r for r in load_jsonl(args.qrels)}

    # Runs from the live server carry the text they returned; prefer it, since a passage
    # id or a non-neuroscience work cannot be looked up in the abstracts Parquet.
    run_texts: dict[str, str] = {}
    for path in args.run:
        for record in load_jsonl(path):
            for key, text in zip(record.get("hits", []), record.get("texts", [])):
                if text and key not in run_texts:
                    run_texts[key] = text

    if args.query_id:
        chosen = [q for q in args.query_id if q in qrels]
    else:
        shared = sorted(set(qrels) & set.intersection(*[set(run) for run in runs]))
        random.seed(args.seed)
        chosen = random.sample(shared, min(args.queries, len(shared)))

    wanted = sorted({key for qid in chosen for run in runs for key in run.get(qid, [])[: args.top]})
    missing = [key for key in wanted if key not in run_texts]
    meta = {}

    if missing:
        rows = duckdb.connect().execute(
            f"""
            SELECT openalex_id, COALESCE(title, ''), COALESCE(publication_year, 0)
            FROM read_parquet('{args.articles}')
            WHERE openalex_id IN (SELECT UNNEST(?))
            """,
            [missing],
        ).fetchall()
        meta = {row[0]: (row[1], row[2]) for row in rows}

    def describe(key: str) -> str:
        if key in run_texts:
            # The composed text starts with the title, then a metadata line.
            return " | ".join(run_texts[key].split("\n")[:2])
        title, year = meta.get(key, ("<not found>", 0))
        return f"{title} ({year})"

    for query_id in chosen:
        record = qrels[query_id]
        gold = set(record.get("gold_article_keys") or [])

        print("=" * args.width)
        print(textwrap.fill(f"[{query_id}] {record['query_text']}", args.width))
        if gold:
            print(f"gold in corpus: {len(gold)}")
        print("=" * args.width)

        for label, run in zip(labels, runs):
            hits = run.get(query_id, [])[: args.top]
            found = sum(1 for key in hits if key in gold)
            suffix = f"  ({found}/{len(hits)} judged relevant)" if gold else ""
            print(f"\n--- {label}{suffix} ---")

            for rank, key in enumerate(hits, start=1):
                mark = "*" if key in gold else " "
                line = f"{mark}{rank:>3}. {describe(key)}"
                print(textwrap.shorten(line, width=args.width, placeholder=" ..."))

        print()

    print("* = in the gold set, where one exists. Unmarked hits are unjudged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
