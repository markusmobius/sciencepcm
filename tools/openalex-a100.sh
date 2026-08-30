#!/usr/bin/env bash
# Prepare and run the second, OpenAlex-specific MCP service on the provisioned A100 box.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${OPENALEX_DATA_ROOT:-$HOME/openalex-data}"
FAST_ROOT="${OPENALEX_FAST_ROOT:-/datadisk}"
SYNC_PYTHON="${SCIENCEPCM_DATA_ROOT:-$HOME/sciencepcm-data}/venvs/sync/bin/python"
LAB_PYTHON="${SCIENCEPCM_DATA_ROOT:-$HOME/sciencepcm-data}/venvs/lab/bin/python"

# Parquet stays on the managed disk: 134 GB read once, sequentially, during the build,
# which is the one access pattern a spinning disk handles well. The index goes on NVMe,
# where Lucene's random reads and segment merges actually need the IOPS.
# Pulled straight into DATA_ROOT, which is where the cloud path lands it. The blob store
# fingerprints files and skips identical ones, so re-running prepare costs nothing and
# picks up a rebuilt digest automatically. Downloading to a temp directory and moving it
# would hide the existing copy and force a full 134 GB transfer every time.
ABSTRACTS="$DATA_ROOT/openalex/abstracts"
DURABLE_INDEX="$DATA_ROOT/index/abstracts-bm25"
FAST_INDEX="$FAST_ROOT/openalex-data/index/abstracts-bm25"
MODEL="$DATA_ROOT/models/openalex-bge"
RETIRED_MODEL="$DATA_ROOT/models/openalex-cross"

if [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]]; then
    INDEX="$FAST_INDEX"
    FAST_AVAILABLE=1
else
    INDEX="$DURABLE_INDEX"
    FAST_AVAILABLE=0
fi

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

# Brings the NVMe copy level with the durable one. Returns non-zero when it cannot, so
# the caller knows a build is still needed.
restore() {
    (( FAST_AVAILABLE )) || return 1
    if index_current "$FAST_INDEX"; then
        echo "index on $FAST_ROOT is current"
        return 0
    fi
    index_current "$DURABLE_INDEX" || return 1

    echo "restoring index from $DURABLE_INDEX to $FAST_INDEX"
    mkdir -p "$FAST_INDEX"
    rsync -a --delete --info=stats2 "$DURABLE_INDEX/" "$FAST_INDEX/"
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
    echo "data root    : $DATA_ROOT ($(free_gb "$DATA_ROOT" 2>/dev/null || echo '?') GB free)"
    echo "abstracts    : $ABSTRACTS [$(size_of "$ABSTRACTS")]"
    if (( FAST_AVAILABLE )); then
        echo "fast index   : $FAST_INDEX [$(size_of "$FAST_INDEX")] ($(free_gb "$FAST_ROOT") GB free on $FAST_ROOT)"
        echo "durable copy : $DURABLE_INDEX [$(size_of "$DURABLE_INDEX")]"
    else
        echo "fast index   : $FAST_ROOT unavailable, serving from $DURABLE_INDEX"
    fi
    echo "reranker     : $MODEL"
    echo "local port   : 8081"
    command -v dotnet >/dev/null || { echo "dotnet is missing" >&2; return 1; }
    [[ -x "$SYNC_PYTHON" ]] || { echo "sync Python missing: $SYNC_PYTHON" >&2; return 1; }
    [[ -x "$LAB_PYTHON" ]] || { echo "lab Python missing: $LAB_PYTHON" >&2; return 1; }
    [[ -n "${CLOUDPDS_CLIENT_HASH:-${legopds_clienthash:-}}" ]] || {
        echo "warning: CLOUDPDS_CLIENT_HASH/legopds_clienthash is unset; the digest cannot be refreshed" >&2
    }
}

prepare() {
    check
    mkdir -p "$DATA_ROOT" "$DATA_ROOT/models" "$DATA_ROOT/index"
    (( FAST_AVAILABLE )) && mkdir -p "$FAST_ROOT/openalex-data/index"

    echo "pulling OpenAlex abstract digest (skips files already present and unchanged)"
    if ! env -u MAXCORES "$SYNC_PYTHON" "$REPO/tools/openalex-cloudstore.py" \
            pull --local "$DATA_ROOT"; then
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
            --out "$DATA_ROOT/models" \
            --reranker BAAI/bge-reranker-v2-m3 \
            --reranker-name openalex-bge
    else
        echo "OpenAlex reranker already present"
    fi

    dotnet build "$REPO/src/OpenAlex.Server/OpenAlex.Server.csproj" \
        -c Release -p:UseGpu=true --nologo

    rm -rf "$RETIRED_MODEL"

    # Three cases, cheapest first. The builder answers "is the index at this path current"
    # for either disk, so restoring the wiped NVMe copy and rebuilding after a new digest
    # are the same decision rather than two services.
    if restore; then
        :
    elif index_current "$INDEX"; then
        echo "index is current"
    else
        mkdir -p "$INDEX"
        dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
            build --input "$ABSTRACTS/*.parquet" --schema openalex \
            --out "$INDEX" --threads 16 --ram-buffer 4096

        # /datadisk is wiped on deallocate, so the NVMe copy is only ever a working copy.
        if (( FAST_AVAILABLE )) && [[ "$INDEX" == "$FAST_INDEX" ]]; then
            echo "mirroring index to the durable disk"
            mkdir -p "$DURABLE_INDEX"
            rsync -a --delete --info=stats2 "$FAST_INDEX/" "$DURABLE_INDEX/"
        fi
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
    restore) restore ;;
    # Anything after 'serve' goes to the server, e.g. serve --citation-prior 2.0
    serve) shift; restore || true; serve "$@" ;;
    *) echo "Usage: $0 [check|prepare|restore|serve [server args...]]" >&2; exit 2 ;;
esac