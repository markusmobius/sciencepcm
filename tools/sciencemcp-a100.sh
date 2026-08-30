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

# Chosen by whether the fast disk is usable, not by whether the index is already there:
# /datadisk is empty after a deallocation and restore() is what fills it.
if [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]]; then
    INDEX_ROOT="$FAST_ROOT/sciencepcm-data/index"
else
    INDEX_ROOT="$DATA_ROOT/index"
fi

ABSTRACTS_INDEX="$INDEX_ROOT/abstracts-bm25"
PASSAGE_INDEX="$INDEX_ROOT/passages-bm25"
CORPUS="${SCIENCEPCM_CORPUS:-$DATA_ROOT/sciencepcm}"
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

# Restoring the wiped NVMe copy and rebuilding after new data are the same question -
# is the index at this path current - so one waterfall answers both. Cheapest branch
# first: keep, restore, rebuild.
prepare() {
    local pairs=(
        "abstracts-bm25|abstracts|$CORPUS/abstracts/*.parquet|"
        "passages-bm25|chunks|$CORPUS/passages-2019-2025/chunks-part-*.parquet|$CORPUS/passages-2019-2025/articles-part-*.parquet"
    )

    for pair in "${pairs[@]}"; do
        IFS='|' read -r name schema glob metadata <<< "$pair"
        local fast="$INDEX_ROOT/$name" durable="$DATA_ROOT/index/$name"
        local metadata_args=()
        [[ -n "$metadata" ]] && metadata_args=(--metadata "$metadata")

        if index_current "$fast" "$schema" "$glob" "$metadata"; then
            echo "$name is current"
            continue
        fi

        if [[ "$fast" != "$durable" ]] && index_current "$durable" "$schema" "$glob" "$metadata"; then
            echo "restoring $name from $durable"
            mkdir -p "$fast"
            rsync -a --delete --info=stats2 "$durable/" "$fast/"
            continue
        fi

        echo "building $name"
        mkdir -p "$fast"
        ( cd "$REPO" && dotnet run --project src/SciencePcm.Lexical -c Release -- build \
            --input "$glob" "${metadata_args[@]}" --schema "$schema" --out "$fast" \
            --threads "$(nproc)" --ram-buffer 2048 )

        if [[ "$fast" != "$durable" ]]; then
            echo "mirroring $name to the durable disk"
            mkdir -p "$durable"
            rsync -a --delete --info=stats2 "$fast/" "$durable/"
        fi
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
