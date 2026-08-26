"""Drive the live MCP server and write a run file, so the deployed system can be judged.

Everything measured so far was offline: Python reranking Python-produced candidates. This
exercises the thing users actually hit - Lucene, the C# pair assembly, the ONNX
cross-encoder on the GPU, over HTTP - and produces a run file in the same shape as the
offline ones, so eval/judge.py can grade them side by side.

Speaks MCP Streamable HTTP directly rather than pulling in a client library: it is three
JSON-RPC calls, and the transport details (session header, SSE framing) are exactly what
a broken deployment gets wrong.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


class McpSession:
    """One MCP session. Not thread-safe; give each worker its own."""

    def __init__(self, endpoint: str, token: str | None, timeout: int) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.session_id: str | None = None
        self.next_id = 1
        self.http = requests.Session()
        self.http.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"

        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "sciencepcm-eval", "version": "1"},
        })
        self._notify("notifications/initialized", {})

    def _post(self, body: dict):
        headers = {"Mcp-Session-Id": self.session_id} if self.session_id else {}
        response = self.http.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)

        returned = response.headers.get("Mcp-Session-Id")
        if returned:
            self.session_id = returned

        response.raise_for_status()
        return response

    def _call(self, method: str, params: dict):
        response = self._post({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        self.next_id += 1

        # A request/response call answers either as plain JSON or as one SSE data frame.
        if "text/event-stream" in (response.headers.get("content-type") or ""):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    break
            else:
                raise RuntimeError("No data frame in SSE response")
        else:
            payload = response.json()

        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["result"]

    def _notify(self, method: str, params: dict) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params})

    def call_tool(self, name: str, arguments: dict):
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        text = "".join(part.get("text", "") for part in result.get("content", []))
        return json.loads(text)


def load_queries(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return records[:limit] if limit else records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True, help="e.g. https://www.sciencemcp.econlabs.org/mcp")
    parser.add_argument("--token", help="Bearer token. Falls back to $SCIENCEPCM_TOKEN.")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tool", default="search_literature")
    parser.add_argument(
        "--id-field",
        default="article_key",
        help="What identifies a hit. Use passage_id for search_full_text, so that two "
        "passages from one paper stay distinct.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--section", default=None, help="Full-text only.")
    parser.add_argument("--fast", action="store_true", help="Ask the server to skip reranking.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Each worker opens its own session.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    import os
    token = args.token or os.getenv("SCIENCEPCM_TOKEN")

    queries = load_queries(args.queries, args.limit)
    print(f"endpoint : {args.endpoint}")
    print(f"tool     : {args.tool}")
    print(f"queries  : {len(queries):,} with {args.workers} session(s)")

    local = threading.local()
    results: dict[str, dict] = {}
    latencies: list[float] = []
    lock = threading.Lock()
    done = 0

    def run(record: dict) -> None:
        nonlocal done
        if not hasattr(local, "session"):
            local.session = McpSession(args.endpoint, token, args.timeout)

        arguments = {"query": record["query_text"], "k": args.k}
        if args.section:
            arguments["section"] = args.section
        if args.fast:
            arguments["fast"] = True

        started = time.perf_counter()
        payload = local.session.call_tool(args.tool, arguments)
        elapsed = (time.perf_counter() - started) * 1000

        hits = []
        scores = []
        texts = []
        for hit in payload.get("results", []):
            hits.append(hit.get(args.id_field) or hit.get("article_key", ""))
            scores.append(hit.get("score", 0.0))
            # Carried along so the judge can grade what the server actually returned,
            # rather than looking the document up again and grading something else.
            texts.append(hit.get("text") or hit.get("abstract_excerpt") or "")

        with lock:
            done += 1
            results[str(record["query_id"])] = {"hits": hits, "scores": scores, "texts": texts}
            latencies.append(elapsed)
            if done % 25 == 0:
                print(f"  {done:,}/{len(queries):,}  ({elapsed:,.0f} ms)")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, queries))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in queries:
            query_id = str(record["query_id"])
            entry = results.get(query_id, {"hits": [], "scores": [], "texts": []})
            stream.write(json.dumps({"query_id": query_id, **entry}) + "\n")

    latencies.sort()
    print()
    print(f"latency p50: {latencies[len(latencies) // 2]:,.0f} ms")
    print(f"latency p95: {latencies[int(len(latencies) * 0.95)]:,.0f} ms")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
