"""Known-item retrieval against a live MCP endpoint, scored on ground-truth DOIs.

The pooled LLM judge in judge.py answers "is this result any good", which needs a model
and costs money. This answers a narrower question exactly - "is THIS paper returned, and
at what rank" - from a DOI written down in advance, so it runs in seconds and gives the
same number every time.

That makes it the regression test for the v3 corpus change. L01 and L02 were absent from
the v2 index entirely, because their publishers deposit no abstract to OpenAlex and v2
required one; the reviews and commentaries written about them were indexed in their
place. If those two come back at rank 1 now, the change did what it was meant to do.

Usage:
    export OPENALEX_TOKEN='...'
    python eval/known_item.py --endpoint https://www.openalexmcp.econlabs.org/mcp \\
        --questions eval/questions-landmark.jsonl --tool search_openalex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_run import McpSession

DOI_PREFIX = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)


def normalise_doi(raw: str | None) -> str:
    return DOI_PREFIX.sub("", (raw or "").strip()).lower()


def rank_of(results: list[dict], expected: str) -> int | None:
    """1-based rank of the expected DOI, or None if it never appears."""
    target = normalise_doi(expected)
    for position, hit in enumerate(results, start=1):
        if normalise_doi(hit.get("doi")) == target:
            return position
    return None


def lookup_argument(session: McpSession, tool: str) -> str:
    """Take the id parameter's name from the server's own schema.

    Hardcoding it is unsafe: get_paper calls it articleKey, get_openalex_work calls it
    openAlexId, and the MCP SDK drops unknown argument names silently, so a mismatch
    reports the paper as missing rather than raising.
    """
    for spec in session._call("tools/list", {})["tools"]:
        if spec["name"] != tool:
            continue
        schema = spec.get("inputSchema", {})
        names = schema.get("required") or list(schema.get("properties", {}))
        if not names:
            raise RuntimeError(f"tool '{tool}' declares no parameters")
        return names[0]
    raise RuntimeError(f"tool '{tool}' is not offered by this server")


def lookup(session: McpSession, tool: str, argument: str, openalex_id: str) -> dict | None:
    """Is the work in the index at all? Separates 'we never ingested it' from 'we ranked
    it badly', which need completely different fixes."""
    try:
        record = session.call_tool(tool, {argument: openalex_id})
    except Exception as error:  # noqa: BLE001
        return {"error": str(error)[:120]}
    return None if record.get("error") else record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--tool", default="search_openalex")
    parser.add_argument("--lookup-tool", default="get_openalex_work",
                        help="Used to check presence when the expected paper is not retrieved.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--token", default=os.environ.get("OPENALEX_TOKEN")
                        or os.environ.get("SCIENCEPCM_TOKEN"))
    parser.add_argument("--fast", action="store_true", help="Skip reranking, to isolate BM25.")
    parser.add_argument("--stages", action="store_true",
                        help="Run each query twice, with and without reranking, and show both "
                             "ranks. A paper BM25 finds but reranking loses needs a different fix "
                             "from one BM25 never retrieves.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out", type=Path, help="Optional JSONL of the full results per query.")
    args = parser.parse_args()

    questions = [json.loads(line) for line in
                 args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]

    session = McpSession(args.endpoint, args.token, args.timeout)
    lookup_arg = lookup_argument(session, args.lookup_tool)
    ranks: list[int | None] = []
    indexed = 0
    records = []

    for question in questions:
        arguments = {"query": question["query_text"], "limit": args.limit}
        if args.fast:
            arguments["fast"] = True

        payload = session.call_tool(args.tool, arguments)
        results = payload.get("results", [])
        rank = rank_of(results, question["expect_doi"])
        ranks.append(rank)

        bm25_rank = None
        if args.stages:
            bm25 = session.call_tool(args.tool, {**arguments, "fast": True})
            bm25_rank = rank_of(bm25.get("results", []), question["expect_doi"])

        top = results[0]["title"][:60] if results else "(nothing returned)"
        stage_note = f"  bm25={bm25_rank if bm25_rank else '-':>3}" if args.stages else ""
        print(f"{question['query_id']}  rank={rank if rank else '-':>3}{stage_note}"
              f"  returned={len(results):>2}  top: {top}")

        present = None
        if rank is None and question.get("expect_openalex_id"):
            present = lookup(session, args.lookup_tool, lookup_arg, question["expect_openalex_id"])
            if present and not present.get("error"):
                indexed += 1
                has_abstract = bool((present.get("abstract") or "").strip())
                print(f"       IN INDEX but not retrieved: {(present.get('title') or '')[:60]}")
                print(f"       abstract={'yes' if has_abstract else 'NO'}  "
                      f"type={present.get('type')}  cited={present.get('cited_by_count')}")
            else:
                print(f"       NOT IN INDEX: {question['expect_openalex_id']}")
        elif rank is not None:
            indexed += 1

        records.append({**question, "rank": rank, "bm25_rank": bm25_rank, "lookup": present,
                        "results": [{"doi": hit.get("doi"), "title": hit.get("title"),
                                     "score": hit.get("score")} for hit in results]})

    total = len(ranks)
    hits = lambda k: sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1 / r for r in ranks if r is not None) / total if total else 0.0

    print()
    print(f"queries : {total}")
    print(f"indexed : {indexed}/{total}  <- corpus coverage, fix the ingest if low")
    for k in (1, 3, args.limit):
        print(f"hit@{k:<4}: {hits(k)}/{total} ({hits(k) / total:.1%})")
    print(f"MRR     : {mrr:.3f}  <- ranking quality, fix retrieval if low while indexed is high")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
