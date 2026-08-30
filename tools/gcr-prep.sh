#!/usr/bin/env bash
# Prepare the A100 box: prerequisites, environments, corpus, models, build, indexes.
#
# Counterpart to tools/sync.ps1 on nerds21. That script produces the corpus and
# uploads it; this one provisions this machine and pulls it down. Idempotent -
# every step checks before acting, so re-running costs little.
#
# Ends with a machine that can serve: both BM25 indexes built, models exported.
# The dense/embedding pipeline is NOT part of this - the served path is BM25 plus
# cross-encoder reranking, and the vectors stay archived in the blob store.
#
#   bash tools/gcr-prep.sh --check     report only
#   bash tools/gcr-prep.sh             provision, pull, build, index, verify
#
# Requires legopds_clienthash (or CLOUDPDS_CLIENT_HASH) in the environment.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/sciencepcm-data}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="$DATA_ROOT/venvs"
CORPUS="$DATA_ROOT/sciencepcm"
MODELS="$DATA_ROOT/models"
INDEXES="$DATA_ROOT/index"
DOTNET_CHANNEL="10.0"

CHECK_ONLY=0
SKIP_PULL=0
FORCE_PULL=0
SKIP_MODELS=0
SKIP_BUILD=0
SKIP_INDEX=0
FORCE_MODELS=0
WITH_MEDCPT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)       CHECK_ONLY=1 ;;
        --skip-pull)   SKIP_PULL=1 ;;
        --force-pull)  FORCE_PULL=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-index)  SKIP_INDEX=1 ;;
        --force-index) echo "--force-index is gone; the index rebuilds itself when the stamp changes" >&2 ;;
        --with-medcpt) WITH_MEDCPT=1 ;;
        -h|--help)     sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)             echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

step()  { printf '\n\033[36m=== %s\033[0m\n' "$1"; }
info()  { printf '  %s\n' "$1"; }
warn()  { printf '  \033[33m%s\033[0m\n' "$1"; }
die()   { printf '\n\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- machine

step "Machine"
info "cores      : $(nproc)"
info "memory     : $(free -g | awk '/^Mem:/ {print $2}') GB"
info "disk (${DATA_ROOT%%/*}/) : $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')B available"

if command -v nvidia-smi >/dev/null 2>&1; then
    info "gpu        : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
    warn "gpu        : nvidia-smi not found. The CUDA build will not work."
fi

if [[ -z "${legopds_clienthash:-}" && -z "${CLOUDPDS_CLIENT_HASH:-}" ]]; then
    die "Set legopds_clienthash (or CLOUDPDS_CLIENT_HASH) before running."
fi

# The blob server writes a MAXCORES warning to stdout, where its client expects
# the port number. Cleared for this process only.
if [[ -n "${MAXCORES:-}" ]]; then
    info "clearing MAXCORES ($MAXCORES) for the blob client"
    unset MAXCORES
fi

# ---------------------------------------------------------------- toolchain

step "Toolchain"

export PATH="$HOME/.dotnet:$HOME/.local/bin:$PATH"

if command -v dotnet >/dev/null 2>&1 && [[ "$(dotnet --version)" == 10.* ]]; then
    info "dotnet     : $(dotnet --version)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "dotnet     : would install SDK $DOTNET_CHANNEL"
else
    info "dotnet     : installing SDK $DOTNET_CHANNEL ..."
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel "$DOTNET_CHANNEL" --install-dir "$HOME/.dotnet"
    grep -q 'HOME/.dotnet' "$HOME/.bashrc" 2>/dev/null || \
        echo 'export PATH="$HOME/.dotnet:$PATH"' >> "$HOME/.bashrc"
    info "dotnet     : $(dotnet --version)"
fi

if command -v uv >/dev/null 2>&1; then
    info "uv         : $(uv --version)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "uv         : would install"
else
    info "uv         : installing ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    info "uv         : $(uv --version)"
fi

# ---------------------------------------------------------------- environments

step "Python environments"

# name -> requirements file, empty means a single git package
make_venv() {
    local name="$1" requirements="$2"
    local venv="$VENVS/$name"
    local python="$venv/bin/python"

    if [[ -x "$python" ]]; then
        info "$(printf '%-10s' "$name") present"
        return
    fi
    if [[ $CHECK_ONLY -eq 1 ]]; then
        warn "$(printf '%-10s' "$name") would create"
        return
    fi

    info "$(printf '%-10s' "$name") creating ..."
    uv venv "$venv" --python 3.12 >/dev/null
    if [[ -n "$requirements" ]]; then
        uv pip install --python "$python" -r "$REPO/$requirements" >/dev/null
    fi
    info "$(printf '%-10s' "$name") ready"
}

make_venv sync ""
make_venv eval "requirements/eval.txt"
make_venv lab  "requirements/lab.txt"

SYNC_PY="$VENVS/sync/bin/python"
EVAL_PY="$VENVS/eval/bin/python"
LAB_PY="$VENVS/lab/bin/python"
CUDA_PY="$VENVS/cuda12/bin/python"

if [[ -x "$SYNC_PY" ]]; then
    if "$SYNC_PY" -c "import RemoteBlobStore" 2>/dev/null; then
        info "RemoteBlobStore present"
    elif [[ $CHECK_ONLY -eq 1 ]]; then
        warn "RemoteBlobStore would install"
    else
        info "RemoteBlobStore installing ..."
        uv pip install --python "$SYNC_PY" \
            "git+https://github.com/markusmobius/newsprinceton-pythoncloud" >/dev/null
        info "RemoteBlobStore ready"
    fi
fi

# ---------------------------------------------------------------- cuda

step "CUDA runtime for ONNX Runtime"

# ORT 1.29's CUDA provider links against CUDA 12 and cuDNN 9. Azure images ship
# CUDA 13 and no cuDNN, so the matching userspace libraries are installed as pip
# wheels instead. The driver is backward compatible, so this is safe.
cuda_lib_path() {
    [[ -x "$CUDA_PY" ]] || return 0
    local site
    site="$("$CUDA_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)" || return 0
    local paths=()
    for dir in "$site"/nvidia/*/lib; do
        [[ -d "$dir" ]] && paths+=("$dir")
    done
    (IFS=:; echo "${paths[*]}")
}

if [[ -x "$CUDA_PY" ]] && [[ -n "$(cuda_lib_path)" ]]; then
    info "already installed"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "would install nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12"
else
    info "installing CUDA 12 + cuDNN 9 wheels ..."
    uv venv "$VENVS/cuda12" --python 3.12 >/dev/null
    uv pip install --python "$CUDA_PY" \
        nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 >/dev/null
    info "installed"
fi

CUDA_LIBS="$(cuda_lib_path)"
if [[ -n "$CUDA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    info "LD_LIBRARY_PATH set for this run"

    # Interactive shells need the same paths for manual dotnet runs.
    if [[ $CHECK_ONLY -eq 0 ]]; then
        cat > "$DATA_ROOT/env.sh" <<EOF
# Source before running SciencePcm.Embed with --gpu:
#   source $DATA_ROOT/env.sh
export PATH="\$HOME/.dotnet:\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
        info "wrote $DATA_ROOT/env.sh"
    fi
fi

# ---------------------------------------------------------------- corpus

step "Corpus"

# A marker file records a completed pull, so re-runs do not re-download.
pull() {
    local cloud="$1" name="$2"
    local marker="$DATA_ROOT/.pulled-$name"

    if [[ -f "$marker" && $FORCE_PULL -eq 0 ]]; then
        info "$(printf '%-22s' "$name") already pulled ($(cat "$marker"))"
        return
    fi
    if [[ $SKIP_PULL -eq 1 ]]; then
        warn "$(printf '%-22s' "$name") skipped"
        return
    fi
    if [[ $CHECK_ONLY -eq 1 ]]; then
        warn "$(printf '%-22s' "$name") would pull from $cloud"
        return
    fi

    info "$(printf '%-22s' "$name") pulling ..."
    "$SYNC_PY" "$REPO/tools/cloudstore.py" pull-dir --cloud "$cloud" --local "$DATA_ROOT"
    local count
    count="$(find "$CORPUS/$name" -type f 2>/dev/null | wc -l)"
    [[ "$count" -gt 0 ]] || die "Pull of $cloud produced no files."
    echo "$count files, $(date -Iseconds)" > "$marker"
    info "$(printf '%-22s' "$name") $count files"
}

mkdir -p "$DATA_ROOT"
pull "sciencepcm/abstracts"            "abstracts"
pull "sciencepcm/passages-2019-2025"   "passages-2019-2025"
pull "sciencepcm/questions"            "questions"

# ---------------------------------------------------------------- models

step "Models"

# The reranker is the only model on the serving path. MedCPT's encoders are only
# needed to revisit dense retrieval, which the LLM judge ruled out, so they are opt-in.
if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "skipped"
else
    if [[ -f "$MODELS/bge-reranker/model.onnx" ]]; then
        info "bge-reranker           already exported"
    elif [[ $CHECK_ONLY -eq 1 ]]; then
        warn "bge-reranker           would export to $MODELS"
    else
        info "checking HuggingFace reachability ..."
        hf_status="$(curl -sSL -o /dev/null -w '%{http_code}' \
            https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/config.json || echo 000)"
        [[ "$hf_status" == "200" ]] || die "HuggingFace returned $hf_status. Sync the models from nerds21 instead."

        info "bge-reranker           exporting (~2.2 GB of weights) ..."
        "$LAB_PY" "$REPO/tools/export_onnx.py" --out "$MODELS" --reranker BAAI/bge-reranker-v2-m3
    fi

    if [[ $WITH_MEDCPT -eq 1 ]]; then
        if [[ -f "$MODELS/medcpt-article/model.onnx" ]]; then
            info "medcpt                 already exported"
        elif [[ $CHECK_ONLY -eq 1 ]]; then
            warn "medcpt                 would export encoders + cross-encoder"
        else
            info "medcpt                 exporting (~900 MB of weights) ..."
            "$LAB_PY" "$REPO/tools/export_onnx.py" --out "$MODELS"
        fi
    fi
fi

# ---------------------------------------------------------------- build

step "Build"

if [[ $SKIP_BUILD -eq 1 ]]; then
    warn "skipped"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "would build with -p:UseGpu=true"
else
    ( cd "$REPO" && dotnet build -c Release -p:UseGpu=true --nologo -v q )
    info "built with the CUDA execution provider"
fi

# ---------------------------------------------------------------- verify

step "Tokenizer parity"

parity="$MODELS/medcpt-article/tokenizer-parity.json"
if [[ $CHECK_ONLY -eq 1 || $SKIP_BUILD -eq 1 ]]; then
    warn "skipped"
elif [[ -f "$parity" ]]; then
    ( cd "$REPO" && dotnet run --project src/SciencePcm.Embed -c Release -p:UseGpu=true -- \
        --model "$MODELS/medcpt-article" --verify-tokenizer "$parity" )
else
    warn "no parity file at $parity"
fi

# ---------------------------------------------------------------- indexes

step "Search indexes"

# Both are BM25 over Lucene. Nothing here needs the GPU; the cross-encoder is only
# used at query time. The index build lives in sciencemcp-a100.sh so the corpus globs
# are defined once, and it rebuilds only when the stamp says the index is stale.
if [[ $SKIP_INDEX -eq 1 || $SKIP_BUILD -eq 1 ]]; then
    warn "search indexes         skipped"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    warn "search indexes         would run: bash tools/sciencemcp-a100.sh prepare"
else
    bash "$REPO/tools/sciencemcp-a100.sh" prepare
fi

# ---------------------------------------------------------------- summary

step "Ready"
cat <<EOF
  corpus  : $CORPUS
  models  : $MODELS
  indexes : $INDEXES
  venvs   : $VENVS

  New shells need the CUDA libraries on the path:
    source $DATA_ROOT/env.sh

  Run the MCP server:
    export SCIENCEPCM_TOKEN=...          # or put it in ~/.bashrc
    dotnet run --project src/SciencePcm.Server -c Release -p:UseGpu=true -- \\
      --index $INDEXES/abstracts-bm25 \\
      --passage-index $INDEXES/passages-bm25 \\
      --cross-encoder $MODELS/bge-reranker \\
      --gpu --urls http://0.0.0.0:8080

  Expose it (relay holds the public TLS endpoint):
    ./tools/mcp-tunnel.sh

  Or install both as services:
    sudo cp deploy/systemd/*.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now mcp-prepare mcp-science-server mcp-tunnel
EOF
