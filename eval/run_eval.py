"""Evaluate a retriever against BioASQ qrels.

Two retrievers are supported:

* ``duckdb-bm25`` - a self-contained lexical baseline built directly from the
  ingest Parquet. It needs none of the .NET index, so it produces the first real
  numbers before any embedding job is run. Treat it as the floor that dense
  retrieval and reranking must beat.
* ``http`` - posts to the MCP search endpoint once it exists.

Passage hits are collapsed to parent articles, because BioASQ judges documents.
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import duckdb

from metrics import MetricAccumulator, format_summary


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[str]:
        """Return article keys, best first."""


class DuckDbBm25Retriever(Retriever):
    def __init__(self, chunks_glob: str, database: Path, rebuild: bool = False) -> None:
        fresh = rebuild or not database.exists()
        if rebuild and database.exists():
            database.unlink()

        self.connection = duckdb.connect(str(database))
        self.connection.execute("INSTALL fts; LOAD fts;")

        if fresh:
            print(f"Building BM25 index from {chunks_glob} ...")
            started = time.perf_counter()
            self.connection.execute(
                f"CREATE TABLE chunks AS SELECT ChunkId, ArticleKey, Text FROM read_parquet('{chunks_glob}')"
            )
            self.connection.execute(
                "PRAGMA create_fts_index('chunks', 'ChunkId', 'Text', stemmer='porter', "
                "stopwords='english', overwrite=1)"
            )
            rows = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            print(f"  indexed {rows:,} chunks in {time.perf_counter() - started:.1f}s")

    def search(self, query: str, k: int) -> list[str]:
        # Over-fetch passages so that collapsing to articles still fills k slots.
        rows = self.connection.execute(
            """
            SELECT ArticleKey, MAX(score) AS best
            FROM (
                SELECT ArticleKey, fts_main_chunks.match_bm25(ChunkId, ?) AS score
                FROM chunks
            )
            WHERE best IS NOT NULL
            GROUP BY ArticleKey
            ORDER BY best DESC
            LIMIT ?
            """,
            [query, k],
        ).fetchall()
        return [str(row[0]) for row in rows]


class HttpRetriever(Retriever):
    def __init__(self, endpoint: str, token: str | None = None) -> None:
        import requests

        self.endpoint = endpoint
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def search(self, query: str, k: int) -> list[str]:
        response = self.session.post(self.endpoint, json={"query": query, "k": k}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [hit["article_key"] for hit in payload.get("results", [])]


def load_qrels(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return records[:limit] if limit else records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--retriever", choices=["duckdb-bm25", "http"], default="duckdb-bm25")
    parser.add_argument("--chunks", help="Glob for chunks-part-*.parquet (duckdb-bm25)")
    parser.add_argument("--database", type=Path, default=Path("bm25.duckdb"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--endpoint", help="MCP search URL (http)")
    parser.add_argument("--token", default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write summary JSON here.")
    args = parser.parse_args()

    if args.retriever == "duckdb-bm25":
        if not args.chunks:
            parser.error("--chunks is required for duckdb-bm25")
        retriever: Retriever = DuckDbBm25Retriever(args.chunks, args.database, args.rebuild)
        name = "duckdb-bm25"
    else:
        if not args.endpoint:
            parser.error("--endpoint is required for http")
        retriever = HttpRetriever(args.endpoint, args.token)
        name = f"http:{args.endpoint}"

    qrels = load_qrels(args.qrels, args.limit)
    print(f"Evaluating {len(qrels):,} queries at k={args.k} ...")

    accumulator = MetricAccumulator()
    for index, record in enumerate(qrels, start=1):
        gold = set(record["gold_article_keys"])
        started = time.perf_counter()
        ranked = retriever.search(record["query_text"], args.k)
        latency_ms = (time.perf_counter() - started) * 1000
        accumulator.add(ranked, gold, latency_ms)

        if index % 50 == 0:
            print(f"  {index:,}/{len(qrels):,}")

    summary = accumulator.summary()
    print()
    print(format_summary(name, summary))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"retriever": name, "k": args.k, "summary": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
