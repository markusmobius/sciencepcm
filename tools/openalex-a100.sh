#!/usr/bin/env bash
# Prepare and run the second, OpenAlex-specific MCP service on the provisioned A100 box.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_ROOT="${MCP_ROOT:-$HOME/mcp}"
FAST_ROOT="${MCP_FAST_ROOT:-/datadisk}"
SYNC_PYTHON="$MCP_ROOT/venvs/sync/bin/python"
LAB_PYTHON="$MCP_ROOT/venvs/lab/bin/python"

# Parquet stays on the managed disk: 134 GB read once, sequentially, during the build,
# which is the one access pattern a spinning disk handles well. It is pulled straight
# into data/, where the cloud path lands it, so the blob store can fingerprint what
# is already there and transfer only what differs.
ABSTRACTS="$MCP_ROOT/data/openalex/abstracts"
MODEL="$MCP_ROOT/models/openalex-bge"

# The index lives only on the NVMe. A durable copy does not fit - the index is larger
# than the free space on the OS disk - so a deallocation that wipes /datadisk means a
# rebuild from Parquet, which prepare does on its own because the stamp goes with it.
INDEX="$FAST_ROOT/index/openalex-abstracts"

STAMP="index-stamp.json"
COMMAND="${1:-prepare}"

has_digest() {
    local directory="$1"
    [[ -f "$directory/openalex-ingest-report.json" ]] &&
    [[ -n "$(find "$directory" -maxdepth 1 -name '*.parquet' -print -quit 2>/dev/null)" ]]
}

has_abstracts() {
    has_digest "$ABSTRACTS"
}

has_index() {
    [[ -f "$INDEX/$STAMP" ]]
}

index_current() {
    dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
        build --input "$ABSTRACTS/*.parquet" --schema openalex --out "$1" --verify \
        >/dev/null 2>&1
}

free_gb() {
    df -B1G --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '
}

size_of() {
    [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"
}

check() {
    echo "service      : OpenAlex MCP"
    echo "host         : $(hostname)"
    echo "root         : $MCP_ROOT ($(free_gb "$MCP_ROOT" 2>/dev/null || echo '?') GB free)"
    echo "abstracts    : $ABSTRACTS [$(size_of "$ABSTRACTS")]"
    echo "index        : $INDEX [$(size_of "$INDEX")] ($(free_gb "$FAST_ROOT" 2>/dev/null || echo '?') GB free on $FAST_ROOT)"
    echo "reranker     : $MODEL"
    echo "local port   : 8081"
    command -v dotnet >/dev/null || { echo "dotnet is missing" >&2; return 1; }
    # The index only ever lives here, so an absent fast disk is a hard failure rather
    # than a quiet fallback onto the OS disk, where it does not fit anyway.
    [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]] || {
        echo "$FAST_ROOT is missing or not writable; the index has nowhere to live" >&2
        echo "  sudo chown \"\$(id -u):\$(id -g)\" $FAST_ROOT" >&2
        return 1
    }
    [[ -x "$SYNC_PYTHON" ]] || { echo "sync Python missing: $SYNC_PYTHON" >&2; return 1; }
    [[ -x "$LAB_PYTHON" ]] || { echo "lab Python missing: $LAB_PYTHON" >&2; return 1; }
    [[ -n "${CLOUDPDS_CLIENT_HASH:-${legopds_clienthash:-}}" ]] || {
        echo "warning: CLOUDPDS_CLIENT_HASH/legopds_clienthash is unset; the digest cannot be refreshed" >&2
    }
}

prepare() {
    check
    mkdir -p "$MCP_ROOT/data" "$MCP_ROOT/models" "$(dirname "$INDEX")"

    echo "pulling OpenAlex abstract digest (skips files already present and unchanged)"
    if ! env -u MAXCORES "$SYNC_PYTHON" "$REPO/tools/openalex-cloudstore.py" \
            pull --local "$MCP_ROOT/data"; then
        # At boot the cloud may be unreachable or the credentials absent. An existing
        # digest is enough to bring the service back up.
        has_abstracts || { echo "pull failed and there is no local digest" >&2; exit 1; }
        echo "pull failed; continuing with the digest already on disk"
    fi

    has_abstracts || {
        echo "pull completed without a report and Parquet shards under $ABSTRACTS" >&2
        exit 1
    }

    if [[ ! -f "$MODEL/model.onnx" || ! -f "$MODEL/tokenizer.onnx" ]]; then
        # bge-reranker-v2-m3 beats ms-marco-MiniLM-L-6-v2 on news-to-paper matching by a
        # wide margin (hit@1 40.0% vs 23.3% over 30 known-item queries) and, being XLM-R
        # based, it is multilingual like the corpus. It costs about 475 ms per query.
        "$LAB_PYTHON" "$REPO/tools/export_onnx.py" \
            --out "$MCP_ROOT/models" \
            --reranker BAAI/bge-reranker-v2-m3 \
            --reranker-name openalex-bge
    else
        echo "OpenAlex reranker already present"
    fi

    dotnet build "$REPO/src/OpenAlex.Server/OpenAlex.Server.csproj" \
        -c Release -p:UseGpu=true --nologo

    if index_current "$INDEX"; then
        echo "index is current"
    else
        # Removed first, not overwritten: Lucene's CREATE mode does not free the old
        # segments until it commits, so an in-place rebuild needs room for two copies.
        rm -rf "$INDEX"
        mkdir -p "$INDEX"
        dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
            build --input "$ABSTRACTS/*.parquet" --schema openalex \
            --out "$INDEX" --threads 16 --ram-buffer 4096
    fi

    echo "OpenAlex A100 preparation complete. Run: bash tools/openalex-a100.sh serve"
}

serve() {
    has_abstracts || { echo "run prepare first" >&2; exit 1; }
    has_index || { echo "run prepare first" >&2; exit 1; }
    [[ -f "$MODEL/model.onnx" && -f "$MODEL/tokenizer.onnx" ]] || { echo "run prepare first" >&2; exit 1; }
    exec dotnet run --project "$REPO/src/OpenAlex.Server" -c Release \
        -p:UseGpu=true -- \
        --index "$INDEX" \
        --cross-encoder "$MODEL" \
        --gpu \
        --urls http://0.0.0.0:8081 \
        "$@"
}

case "$COMMAND" in
    check) check ;;
    prepare) prepare ;;
    # Anything after 'serve' goes to the server, e.g. serve --citation-prior 2.0
    serve) shift; serve "$@" ;;
    *) echo "Usage: $0 [check|prepare|serve [server args...]]" >&2; exit 2 ;;
esac