"""Throwaway: response shape of the older academic MCP endpoint."""
import json
import os
import sys

sys.path.insert(0, "eval")
from mcp_run import McpSession, compose_text

token = os.environ["OPENALEX_TOKEN"]
session = McpSession("https://www.academic.econlabs.org/mcp", token, 120)

payload = session.call_tool("search_works", {
    "query": "What evidence links social media use to adolescent mental health outcomes?",
    "limit": 3,
})

print("top-level keys :", list(payload) if isinstance(payload, dict) else f"list[{len(payload)}]")
results = payload.get("results") if isinstance(payload, dict) else payload
print("results type   :", type(results).__name__, "count:", len(results or []))

if results:
    first = results[0]
    print("\nhit keys:")
    for key, value in first.items():
        shown = str(value)
        print(f"  {key:<22} {shown[:70]}")
    print("\ncompose_text would produce:")
    print("  " + compose_text(first)[:400].replace("\n", "\n  "))
