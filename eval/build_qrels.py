"""Build retrieval qrels for the SciencePCM index from BioASQ Task B gold PMIDs.

Gold judgements come from the ``pmids`` field of ``queries.jsonl``. ``answers.jsonl``
and ``evidence.jsonl`` are never read, so the corpus rule against answer/evidence
leakage holds by construction.

BioASQ spans all of PubMed, so most gold documents fall outside any given index
subset. A query is only scoreable when at least one gold PMID is actually present;
scoring the rest would report a recall ceiling rather than retrieval quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def load_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_index_pmids(articles_glob: str) -> dict[str, str]:
    """Map PMID -> ArticleKey for every article present in the index."""
    rows = duckdb.sql(
        f"""
        SELECT Pmid, ArticleKey
        FROM read_parquet('{articles_glob}')
        WHERE Pmid IS NOT NULL AND Pmid <> ''
        """
    ).fetchall()
    return {str(pmid): str(key) for pmid, key in rows}


def load_neuro_ids(questions_path: Path | None) -> set[str] | None:
    if questions_path is None:
        return None
    ids = set()
    with questions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.add(record.get("query_id"))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", required=True, type=Path, help="bioasq-full/queries.jsonl")
    parser.add_argument("--articles", required=True, help="Glob for articles-part-*.parquet")
    parser.add_argument("--out", required=True, type=Path, help="Destination qrels.jsonl")
    parser.add_argument(
        "--neuroscience-questions",
        type=Path,
        default=None,
        help="Optional v0.2 questions JSONL; restricts qrels to audited neuroscience queries.",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    pmid_to_key = load_index_pmids(args.articles)
    neuro_ids = load_neuro_ids(args.neuroscience_questions)

    kept: list[dict] = []
    total_gold = 0
    retrievable_gold = 0
    dropped_no_overlap = 0
    dropped_not_neuro = 0

    for query in queries:
        if neuro_ids is not None and query.get("query_id") not in neuro_ids:
            dropped_not_neuro += 1
            continue

        gold_pmids = [str(p) for p in query.get("pmids") or []]
        total_gold += len(gold_pmids)

        present = {p: pmid_to_key[p] for p in gold_pmids if p in pmid_to_key}
        retrievable_gold += len(present)

        if not present:
            dropped_no_overlap += 1
            continue

        kept.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "question_type": query.get("question_type"),
                "gold_article_keys": sorted(present.values()),
                "gold_pmids_in_index": sorted(present.keys()),
                "gold_pmids_total": len(gold_pmids),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in kept:
            stream.write(json.dumps(record) + "\n")

    considered = len(queries) - dropped_not_neuro
    print(f"Queries in source        : {len(queries):,}")
    if neuro_ids is not None:
        print(f"Dropped (not neuro)      : {dropped_not_neuro:,}")
    print(f"Considered               : {considered:,}")
    print(f"Dropped (no gold in idx) : {dropped_no_overlap:,}")
    print(f"Scoreable queries        : {len(kept):,}")
    print(f"Gold docs total          : {total_gold:,}")
    print(f"Gold docs in index       : {retrievable_gold:,} ({retrievable_gold / max(1, total_gold):.1%})")
    print(f"Articles indexed         : {len(pmid_to_key):,}")
    print(f"Wrote                    : {args.out}")

    if len(kept) < 50:
        print()
        print("WARNING: fewer than 50 scoreable queries. Metrics will be too noisy to")
        print("compare retrieval configurations. Widen the index subset before tuning.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
