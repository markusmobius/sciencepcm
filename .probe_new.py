"""Throwaway: compare hit key spaces so the judge can pool across both servers."""
import json
import os
import sys

sys.path.insert(0, "eval")
from mcp_run import McpSession

token = os.environ["OPENALEX_TOKEN"]
q = "What evidence links social media use to adolescent mental health outcomes?"

session = McpSession("https://www.openalexmcp.econlabs.org/mcp", token, 120)
payload = session.call_tool("search_openalex", {"query": q, "limit": 3})
results = payload.get("results") if isinstance(payload, dict) else payload

print("new server top-level:", list(payload) if isinstance(payload, dict) else f"list[{len(payload)}]")
print("\nhit keys:")
for key, value in results[0].items():
    print(f"  {key:<22} {str(value)[:70]}")
