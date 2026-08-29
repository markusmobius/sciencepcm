#!/usr/bin/env bash
# Run the neuroscience MCP service on the provisioned A100 box.
#
# Preparation (corpus, models, indexes) lives in gcr-prep.sh; this owns the paths and the
# serving arguments, so the systemd unit and the README no longer keep separate copies of
# them that drift apart.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SCIENCEPCM_DATA_ROOT:-$HOME/sciencepcm-data}"
FAST_ROOT="${SCIENCEPCM_FAST_ROOT:-/datadisk}"
MODEL="$DATA_ROOT/models/bge-reranker"
PORT="${SCIENCEPCM_PORT:-8080}"

# /datadisk is local NVMe and ephemeral; the managed disk holds the durable copy that
# datadisk-restore.sh syncs across at boot.
if [[ -d "$FAST_ROOT/sciencepcm-data/index" ]]; then
    INDEX_ROOT="$FAST_ROOT/sciencepcm-data/index"
else
    INDEX_ROOT="$DATA_ROOT/index"
fi

ABSTRACTS_INDEX="$INDEX_ROOT/abstracts-bm25"
PASSAGE_INDEX="$INDEX_ROOT/passages-bm25"
COMMAND="${1:-serve}"

size_of() {
    [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"
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
    [[ -d "$ABSTRACTS_INDEX" ]] || { echo "no abstracts index; run: bash tools/gcr-prep.sh" >&2; return 1; }
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
    prepare) exec bash "$REPO/tools/gcr-prep.sh" "${@:2}" ;;
    restore) exec bash "$REPO/tools/datadisk-restore.sh" ;;
    # Anything after 'serve' goes to the server, e.g. serve --rerank-candidates 200
    serve) shift; serve "$@" ;;
    *) echo "Usage: $0 [check|prepare|restore|serve [server args...]]" >&2; exit 2 ;;
esac
