"""Fuse run files with Reciprocal Rank Fusion.

BM25 and dense retrieval fail differently: lexical search misses paraphrase, dense
search misses rare entities that never appeared in training. Their unions therefore
recall more than either alone, and recall is what bounds the reranker.

RRF combines by RANK rather than score, which matters because BM25 scores are
unbounded and cosine scores sit in a narrow band - any weighted score sum would be
dominated by whichever system happens to have the wider range.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_run(path: Path) -> dict[str, list[str]]:
    runs = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            runs[str(record["query_id"])] = record["hits"]
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, help="Repeat for each system.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, default=100, help="Results to keep per query.")
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF damping constant. 60 is the value from the original paper.",
    )
    args = parser.parse_args()

    if len(args.run) < 2:
        parser.error("Pass --run at least twice; fusing one system is a copy.")

    runs = [load_run(path) for path in args.run]
    for path, run in zip(args.run, runs):
        print(f"{path.name}: {len(run):,} queries")

    query_ids = set(runs[0])
    for run in runs[1:]:
        query_ids &= set(run)
    print(f"shared queries: {len(query_ids):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for query_id in sorted(query_ids):
            fused: dict[str, float] = defaultdict(float)
            for run in runs:
                for rank, key in enumerate(run[query_id], start=1):
                    fused[key] += 1.0 / (args.rrf_k + rank)

            ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[: args.k]
            stream.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "hits": [key for key, _ in ranked],
                        "scores": [score for _, score in ranked],
                    }
                )
                + "\n"
            )

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
