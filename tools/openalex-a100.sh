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
ABSTRACTS="$DATA_ROOT/abstracts"
DURABLE_INDEX="$DATA_ROOT/index/abstracts-bm25"
FAST_INDEX="$FAST_ROOT/openalex-data/index/abstracts-bm25"
MODEL="$DATA_ROOT/models/openalex-bge"
RETIRED_MODEL="$DATA_ROOT/models/openalex-cross"
PULL_ROOT="$DATA_ROOT/.abstracts-pull"

if [[ -d "$FAST_ROOT" && -w "$FAST_ROOT" ]]; then
    INDEX="$FAST_INDEX"
    FAST_AVAILABLE=1
else
    INDEX="$DURABLE_INDEX"
    FAST_AVAILABLE=0
fi

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
        echo "CLOUDPDS_CLIENT_HASH/legopds_clienthash is unset" >&2
        return 1
    }
}

# The v3 digest roughly doubled the corpus, so a v2 index left on either disk is both
# stale and large enough to starve the rebuild. This clears both locations at once.
clean() {
    local targets=("$ABSTRACTS" "$DATA_ROOT/index" "$FAST_ROOT/openalex-data/index"
                   "$RETIRED_MODEL" "$PULL_ROOT")

    echo "would remove:"
    for target in "${targets[@]}"; do
        [[ -e "$target" ]] && echo "  $target [$(size_of "$target")]"
    done

    if [[ "${1:-}" != "--yes" ]]; then
        echo
        echo "nothing removed. Re-run with: $0 clean --yes"
        return 0
    fi

    for target in "${targets[@]}"; do
        [[ -e "$target" ]] || continue
        echo "removing $target"
        rm -rf "$target"
    done

    echo "cleaned. $(free_gb "$DATA_ROOT") GB free on $DATA_ROOT$( (( FAST_AVAILABLE )) && echo ", $(free_gb "$FAST_ROOT") GB free on $FAST_ROOT")"
}

prepare() {
    check
    mkdir -p "$DATA_ROOT" "$DATA_ROOT/models" "$DATA_ROOT/index"
    (( FAST_AVAILABLE )) && mkdir -p "$FAST_ROOT/openalex-data/index"

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

    if ! has_index; then
        local available
        available="$(free_gb "$(dirname "$(dirname "$INDEX")")" 2>/dev/null || echo 0)"
        if (( available < 500 )); then
            echo "WARNING: only ${available} GB free where the index will be built."
            echo "         v3 indexes ~485M works; budget 400-450 GB plus merge headroom."
            echo "         Run '$0 clean --yes' to drop the stale v2 index first."
        fi

        rm -rf "$INDEX"
        mkdir -p "$INDEX"
        dotnet run --project "$REPO/src/OpenAlex.Index" -c Release -- \
            build --input "$ABSTRACTS/*.parquet" --schema openalex \
            --require-body \
            --out "$INDEX" --threads 16 --ram-buffer 4096
        touch "$INDEX_COMPLETE"
    else
        echo "OpenAlex Lucene index already present"
    fi

    # /datadisk is wiped when the VM deallocates, so the NVMe copy is only ever a working
    # copy. datadisk-restore.service syncs this durable one back at boot.
    if (( FAST_AVAILABLE )) && [[ "$INDEX" == "$FAST_INDEX" ]]; then
        echo "mirroring index to the durable disk"
        mkdir -p "$DURABLE_INDEX"
        rsync -a --delete --info=stats2 "$FAST_INDEX/" "$DURABLE_INDEX/"
    fi

    echo "OpenAlex A100 preparation complete. Run: bash tools/openalex-a100.sh serve"
}

serve() {
    has_abstracts || { echo "run prepare first" >&2; exit 1; }
    has_index || { echo "run prepare first" >&2; exit 1; }
    [[ -f "$MODEL/model.onnx" && -f "$MODEL/tokenizer.onnx" ]] || { echo "run prepare first" >&2; exit 1; }
    exec dotnet run --project "$REPO/src/OpenAlex.Server" -c Release \
        -p:UseGpu=true --no-build -- \
        --index "$INDEX" \
        --cross-encoder "$MODEL" \
        --gpu \
        --urls http://0.0.0.0:8081 \
        "$@"
}

case "$COMMAND" in
    check) check ;;
    clean) clean "${2:-}" ;;
    prepare) prepare ;;
    # Anything after 'serve' goes to the server, e.g. serve --max-doc-freq-ratio 0.01
    serve) shift; serve "$@" ;;
    *) echo "Usage: $0 [check|clean [--yes]|prepare|serve [server args...]]" >&2; exit 2 ;;
esac