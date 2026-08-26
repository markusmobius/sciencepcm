#!/usr/bin/env bash
# Prepare and run the second, OpenAlex-specific MCP service on the provisioned A100 box.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${OPENALEX_DATA_ROOT:-$HOME/openalex-data}"
SYNC_PYTHON="${SCIENCEPCM_DATA_ROOT:-$HOME/sciencepcm-data}/venvs/sync/bin/python"
LAB_PYTHON="${SCIENCEPCM_DATA_ROOT:-$HOME/sciencepcm-data}/venvs/lab/bin/python"
ABSTRACTS="$DATA_ROOT/abstracts"
INDEX="$DATA_ROOT/index/abstracts-bm25"
MODEL="$DATA_ROOT/models/openalex-cross"
PULL_ROOT="$DATA_ROOT/.abstracts-pull"
INDEX_COMPLETE="$INDEX/.openalex-index-complete"
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
    [[ -f "$INDEX_COMPLETE" ]] &&
        { [[ -f "$INDEX/segments.gen" ]] ||
            [[ -n "$(find "$INDEX" -maxdepth 1 -name 'segments_*' -print -quit 2>/dev/null)" ]]; }
}

check() {
    echo "service      : OpenAlex MCP"
    echo "host         : $(hostname)"
    echo "data root    : $DATA_ROOT"
    echo "abstracts    : $ABSTRACTS"
    echo "index        : $INDEX"
    echo "reranker     : $MODEL"
    echo "local port   : 8081"
    command -v dotnet >/dev/null || { echo "dotnet is missing" >&2; return 1; }
    [[ -x "$SYNC_PYTHON" ]] || { echo "sync Python missing: $SYNC_PYTHON" >&2; return 1; }
    [[ -x "$LAB_PYTHON" ]] || { echo "lab Python missing: $LAB_PYTHON" >&2; return 1; }
    [[ -n "${CLOUDPDS_CLIENT_HASH:-${legopds_clienthash:-}}" ]] || {
        echo "CLOUDPDS_CLIENT_HASH/legopds_clienthash is unset" >&2
        return 1
    }
}

prepare() {
    check
    mkdir -p "$DATA_ROOT" "$DATA_ROOT/models" "$DATA_ROOT/index"

    if ! has_abstracts; then
        echo "pulling OpenAlex abstract digest"
        rm -rf "$ABSTRACTS" "$PULL_ROOT"
        env -u MAXCORES "$SYNC_PYTHON" "$REPO/tools/openalex-cloudstore.py" \
            pull --local "$PULL_ROOT"

        pulled="$PULL_ROOT/openalex/abstracts"
        has_digest "$pulled" || {
            echo "OpenAlex pull completed without a report and Parquet shards" >&2
            exit 1
        }
        mv "$pulled" "$ABSTRACTS"
        rm -rf "$PULL_ROOT"
    else
        echo "OpenAlex abstract digest already present"
    fi

    if [[ ! -f "$MODEL/model.onnx" ]]; then
        "$LAB_PYTHON" "$REPO/tools/export_onnx.py" \
            --out "$DATA_ROOT/models" \
            --cross-only \
            --cross-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --cross-name openalex-cross
    else
        echo "OpenAlex reranker already present"
    fi

    dotnet build "$REPO/src/OpenAlex.Server/OpenAlex.Server.csproj" \
        -c Release -p:UseGpu=true --nologo

    if ! has_index; then
        rm -rf "$INDEX"
        mkdir -p "$INDEX"
        dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
            build --input "$ABSTRACTS/*.parquet" --schema openalex \
            --out "$INDEX" --threads 16 --ram-buffer 4096
        touch "$INDEX_COMPLETE"
    else
        echo "OpenAlex Lucene index already present"
    fi

    echo "OpenAlex A100 preparation complete. Run: bash tools/openalex-a100.sh serve"
}

serve() {
    has_abstracts || { echo "run prepare first" >&2; exit 1; }
    has_index || { echo "run prepare first" >&2; exit 1; }
    [[ -f "$MODEL/model.onnx" ]] || { echo "run prepare first" >&2; exit 1; }
    exec dotnet run --project "$REPO/src/OpenAlex.Server" -c Release \
        -p:UseGpu=true --no-build -- \
        --index "$INDEX" \
        --cross-encoder "$MODEL" \
        --gpu \
        --urls http://0.0.0.0:8081
}

case "$COMMAND" in
    check) check ;;
    prepare) prepare ;;
    serve) serve ;;
    *) echo "Usage: $0 [check|prepare|serve]" >&2; exit 2 ;;
esac