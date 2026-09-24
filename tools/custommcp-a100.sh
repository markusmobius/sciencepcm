#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_ROOT="${MCP_ROOT:-$HOME/mcp}"
FAST_ROOT="${MCP_FAST_ROOT:-/datadisk}"
CONFIG="${CUSTOMMCP_CONFIG:-$REPO/custom_mcps/pds.json}"
DATA_ROOT="$MCP_ROOT/data/custommcp"
INDEX_ROOT="$FAST_ROOT/index/custommcp"
MODEL="$MCP_ROOT/models/bge-reranker"
LAB_PYTHON="$MCP_ROOT/venvs/lab/bin/python"
PORT="${CUSTOMMCP_PORT:-8083}"
COMMAND="${1:-serve}"
[[ $# -eq 0 ]] || shift

prerequisites() {
    echo "service        : CustomMcp"
    echo "configuration  : $CONFIG"
    echo "data root      : $DATA_ROOT"
    echo "index root     : $INDEX_ROOT"
    echo "shared reranker: $MODEL"
    echo "local port     : $PORT"
    command -v dotnet >/dev/null || { echo "dotnet is missing" >&2; return 1; }
    [[ -f "$CONFIG" ]] || { echo "missing PDS collection config: $CONFIG" >&2; return 1; }
    [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]] || {
        echo "$FAST_ROOT is missing or not writable; refusing to place indexes on the OS disk" >&2
        return 1
    }
}

check() {
    prerequisites
    ( cd "$REPO" && dotnet run --project src/CustomMcp.Ingest -c Release -- \
        check --config "$CONFIG" --index-root "$INDEX_ROOT" )
    [[ -f "$MODEL/model.onnx" && -f "$MODEL/tokenizer.onnx" ]] || {
        echo "shared reranker is missing; run: bash tools/custommcp-a100.sh prepare" >&2
        return 1
    }
}

prepare() {
    prerequisites
    mkdir -p "$DATA_ROOT" "$INDEX_ROOT" "$MCP_ROOT/models"
    unset MAXCORES
    ( cd "$REPO" && dotnet run --project src/CustomMcp.Ingest -c Release -- \
        prepare --config "$CONFIG" --data-root "$DATA_ROOT" --index-root "$INDEX_ROOT" \
        --threads "$(nproc)" "$@" )

    if [[ ! -f "$MODEL/model.onnx" || ! -f "$MODEL/tokenizer.onnx" ]]; then
        [[ -x "$LAB_PYTHON" ]] || { echo "missing lab environment; run tools/gcr-prep.sh first" >&2; return 1; }
        "$LAB_PYTHON" "$REPO/tools/export_onnx.py" \
            --out "$MCP_ROOT/models" --reranker BAAI/bge-reranker-v2-m3
        if [[ -f "$MODEL/tokenizer-parity.json" ]]; then
            ( cd "$REPO" && dotnet run --project src/SciencePcm.Embed -c Release -- \
                --model "$MODEL" --verify-pairs "$MODEL/tokenizer-parity.json" )
        fi
    else
        echo "shared reranker already present"
    fi
}

serve() {
    prerequisites
    [[ -f "$MODEL/model.onnx" && -f "$MODEL/tokenizer.onnx" ]] || {
        echo "shared reranker is missing; run: bash tools/custommcp-a100.sh prepare" >&2
        return 1
    }
    exec dotnet run --project "$REPO/src/CustomMcp.Server" -c Release -p:UseGpu=true -- \
        --config "$CONFIG" --index-root "$INDEX_ROOT" --cross-encoder "$MODEL" \
        --gpu --urls "http://127.0.0.1:$PORT" "$@"
}

case "$COMMAND" in
    check) check ;;
    prepare) prepare "$@" ;;
    serve) serve "$@" ;;
    *) echo "Usage: $0 [check|prepare [ingest args...]|serve [server args...]]" >&2; exit 2 ;;
esac