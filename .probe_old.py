"""Throwaway: what does the older academic MCP endpoint expose?"""
import json
import os
import sys

sys.path.insert(0, "eval")
from mcp_run import McpSession

token = os.environ["OPENALEX_TOKEN"]
for url in ("https://www.academic.econlabs.org/mcp",
            "https://www.openalexmcp.econlabs.org/mcp"):
    print("=" * 70)
    print(url)
    try:
        session = McpSession(url, token, 60)
    except Exception as exc:                        # noqa: BLE001
        print(f"  connect failed: {exc}")
        continue
    for spec in session._call("tools/list", {})["tools"]:
        schema = spec.get("inputSchema", {})
        props = list(schema.get("properties", {}))
        print(f"  {spec['name']:<24} {props}")
