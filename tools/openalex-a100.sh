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
COMMAND="${1:-prepare}"

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

    if [[ ! -f "$ABSTRACTS/openalex-ingest-report.json" ]]; then
        env -u MAXCORES "$SYNC_PYTHON" "$REPO/tools/openalex-cloudstore.py" \
            pull --local "$ABSTRACTS"
    else
        echo "abstract digest already present; remove it to pull again"
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

    if [[ ! -f "$INDEX/segments.gen" && -z "$(find "$INDEX" -maxdepth 1 -name 'segments_*' -print -quit 2>/dev/null)" ]]; then
        mkdir -p "$INDEX"
        dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
            build --input "$ABSTRACTS/*.parquet" --schema openalex \
            --out "$INDEX" --threads 16 --ram-buffer 4096
    else
        echo "OpenAlex Lucene index already present"
    fi

    echo "OpenAlex A100 preparation complete. Run: bash tools/openalex-a100.sh serve"
}

serve() {
    [[ -f "$ABSTRACTS/openalex-ingest-report.json" ]] || { echo "run prepare first" >&2; exit 1; }
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