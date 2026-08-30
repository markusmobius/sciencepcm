#!/usr/bin/env bash
# Run the neuroscience MCP service on the provisioned A100 box.
#
# Preparation (corpus, models, indexes) lives in gcr-prep.sh; this owns the paths and the
# serving arguments, so the systemd unit and the README no longer keep separate copies of
# them that drift apart.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_ROOT="${MCP_ROOT:-$HOME/mcp}"
FAST_ROOT="${MCP_FAST_ROOT:-/datadisk}"
SYNC_PYTHON="$MCP_ROOT/venvs/sync/bin/python"
LAB_PYTHON="$MCP_ROOT/venvs/lab/bin/python"
CORPUS="$MCP_ROOT/data/sciencepcm"
# The same bge-reranker-v2-m3 export both services use.
MODEL="$MCP_ROOT/models/bge-reranker"
PORT="${SCIENCEPCM_PORT:-8080}"

# The index lives only on the NVMe. A durable copy of the OpenAlex index does not fit on
# the OS disk, so neither service keeps one and both rebuild after a deallocation - the
# stamp goes with the wiped disk, so prepare notices on its own.
INDEX_ROOT="$FAST_ROOT/index"

ABSTRACTS_INDEX="$INDEX_ROOT/science-abstracts"
PASSAGE_INDEX="$INDEX_ROOT/science-passages"
COMMAND="${1:-serve}"

size_of() {
    [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"
}

index_current() {
    local out="$1" schema="$2" glob="$3" metadata="${4:-}"
    local metadata_args=()
    [[ -n "$metadata" ]] && metadata_args=(--metadata "$metadata")
    ( cd "$REPO" && dotnet run --project src/SciencePcm.Lexical -c Release -- \
        build --input "$glob" "${metadata_args[@]}" --schema "$schema" --out "$out" --verify \
        >/dev/null 2>&1 )
}

# Restoring and rebuilding collapse into one question - is the index at this path
# current - now that there is only one path.
prepare() {
    check
    mkdir -p "$MCP_ROOT/data" "$MCP_ROOT/models" "$INDEX_ROOT"

    for name in abstracts passages-2019-2025 questions; do
        echo "pulling sciencepcm/$name (transfers only what differs)"
        "$SYNC_PYTHON" "$REPO/tools/cloudstore.py" pull-dir \
            --cloud "sciencepcm/$name" --local "$MCP_ROOT/data"
    done

    if [[ ! -f "$MODEL/model.onnx" || ! -f "$MODEL/tokenizer.onnx" ]]; then
        "$LAB_PYTHON" "$REPO/tools/export_onnx.py" \
            --out "$MCP_ROOT/models" \
            --reranker BAAI/bge-reranker-v2-m3
    else
        echo "reranker already present"
    fi

    local pairs=(
        "science-abstracts|abstracts|$CORPUS/abstracts/*.parquet|"
        "science-passages|chunks|$CORPUS/passages-2019-2025/chunks-part-*.parquet|$CORPUS/passages-2019-2025/articles-part-*.parquet"
    )

    for pair in "${pairs[@]}"; do
        IFS='|' read -r name schema glob metadata <<< "$pair"
        local out="$INDEX_ROOT/$name"
        local metadata_args=()
        [[ -n "$metadata" ]] && metadata_args=(--metadata "$metadata")

        if index_current "$out" "$schema" "$glob" "$metadata"; then
            echo "$name is current"
            continue
        fi

        echo "building $name"
        # Removed first, not overwritten: Lucene's CREATE mode does not free the old
        # segments until it commits, so an in-place rebuild needs room for two copies.
        rm -rf "$out"
        mkdir -p "$out"
        ( cd "$REPO" && dotnet run --project src/SciencePcm.Lexical -c Release -- build \
            --input "$glob" "${metadata_args[@]}" --schema "$schema" --out "$out" \
            --threads "$(nproc)" --ram-buffer 2048 )
    done
}

stamp_of() {
    local stamp="$1/index-stamp.json"
    [[ -f "$stamp" ]] && tr -d ' \n' < "$stamp" | sed 's/.*"SchemaVersion":\([0-9]*\).*/schema v\1/' || echo "no stamp - rebuild needed"
}

check() {
    echo "service        : ScienceMCP"
    echo "host           : $(hostname)"
    echo "abstracts index: $ABSTRACTS_INDEX [$(size_of "$ABSTRACTS_INDEX")] $(stamp_of "$ABSTRACTS_INDEX")"
    echo "passage index  : $PASSAGE_INDEX [$(size_of "$PASSAGE_INDEX")] $(stamp_of "$PASSAGE_INDEX")"
    echo "reranker       : $MODEL"
    echo "local port     : $PORT"
    command -v dotnet >/dev/null || { echo "dotnet is missing" >&2; return 1; }
    # The index only ever lives here, so an absent fast disk is a hard failure rather
    # than a quiet fallback onto the OS disk.
    [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]] || {
        echo "$FAST_ROOT is missing or not writable; the index has nowhere to live" >&2
        echo "  sudo chown \"\$(id -u):\$(id -g)\" $FAST_ROOT" >&2
        return 1
    }
    [[ -d "$ABSTRACTS_INDEX" ]] || { echo "no abstracts index; run: bash tools/sciencemcp-a100.sh prepare" >&2; return 1; }
    [[ -f "$MODEL/model.onnx" ]] || { echo "no reranker; run: bash tools/gcr-prep.sh" >&2; return 1; }
}

serve() {
    check
    local passage_args=()
    [[ -d "$PASSAGE_INDEX" ]] && passage_args=(--passage-index "$PASSAGE_INDEX")

    exec dotnet run --project "$REPO/src/SciencePcm.Server" -c Release \
        -p:UseGpu=true -- \
        --index "$ABSTRACTS_INDEX" \
        "${passage_args[@]}" \
        --cross-encoder "$MODEL" \
        --gpu \
        --urls "http://0.0.0.0:$PORT" \
        "$@"
}

case "$COMMAND" in
    check) check ;;
    prepare) prepare ;;
    # Anything after 'serve' goes to the server, e.g. serve --rerank-candidates 200
    serve) shift; serve "$@" ;;
    *) echo "Usage: $0 [check|prepare|serve [server args...]]" >&2; exit 2 ;;
esac
