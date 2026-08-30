#!/usr/bin/env bash
# Prepare the A100 box: prerequisites, Python environments, CUDA runtime, build.
#
# The machine only. Each service pulls its own data, exports the shared reranker and
# builds its own index in its own prepare - see sciencemcp-a100.sh and openalex-a100.sh.
# Idempotent: every step checks before acting, so re-running costs little.
#
# Counterpart to tools/sync.ps1 on nerds21. That script produces the corpus and
# uploads it; this one provisions this machine and pulls it down. Idempotent -
# every step checks before acting, so re-running costs little.
#
# Ends with a machine that can serve: both BM25 indexes built, models exported.
# The dense/embedding pipeline is NOT part of this - the served path is BM25 plus
# cross-encoder reranking, and the vectors stay archived in the blob store.
#
# Offers to install the systemd units, asking for each server's token. Tokens already
# present in the installed units are reused, so re-running never re-asks.
#
#   bash tools/gcr-prep.sh --check     report only
#   bash tools/gcr-prep.sh             provision, pull, build, index, verify
#
# Requires legopds_clienthash (or CLOUDPDS_CLIENT_HASH) in the environment.

set -euo pipefail

MCP_ROOT="${MCP_ROOT:-$HOME/mcp}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="$MCP_ROOT/venvs"
MODELS="$MCP_ROOT/models"
TEMP_NETCORE="${TEMP_NETCORE:-$HOME/temp_netcore}"
DOTNET_CHANNEL="10.0"

CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)       CHECK_ONLY=1 ;;
        # Data, models and indexes moved into each service's prepare. --skip-build went
        # with --no-build: dotnet run rebuilds anyway, so skipping here only hides errors.
        --skip-pull|--force-pull|--skip-models|--skip-index|--force-index|--with-medcpt|--skip-build)
                       echo "$1 is gone; see tools/{sciencemcp,openalex}-a100.sh prepare" >&2 ;;
        -h|--help)     sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
info "disk (${MCP_ROOT%%/*}/) : $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')B available"

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
fi

# Written even without CUDA: every serve script and systemd unit sources this file, so
# a missing one stops the servers starting rather than merely losing GPU support.
if [[ $CHECK_ONLY -eq 0 ]]; then
    mkdir -p "$TEMP_NETCORE"
    cat > "$MCP_ROOT/env.sh" <<EOF
# Sourced by tools/*-a100.sh and by every systemd unit.
#   source $MCP_ROOT/env.sh
export PATH="\$HOME/.dotnet:\$HOME/.local/bin:\$PATH"
export TEMP_NETCORE="$TEMP_NETCORE"
EOF
    if [[ -n "$CUDA_LIBS" ]]; then
        cat >> "$MCP_ROOT/env.sh" <<EOF
export LD_LIBRARY_PATH="$CUDA_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
    else
        warn "no CUDA libraries found; env.sh written without LD_LIBRARY_PATH"
    fi
    info "wrote $MCP_ROOT/env.sh"
fi

# ---------------------------------------------------------------- build

step "Build"

if [[ $CHECK_ONLY -eq 1 ]]; then
    warn "would build with -p:UseGpu=true"
else
    ( cd "$REPO" && dotnet build -c Release -p:UseGpu=true --nologo -v q )
    info "built with the CUDA execution provider"
fi

# ---------------------------------------------------------------- services

step "Services"

SYSTEMD_DIR=/etc/systemd/system
# mcp-console.service is missing on purpose: it runs on the relay, not on this box.
UNITS=(mcp-prepare.service mcp-science-server.service mcp-openalex-server.service mcp-tunnel.service)

TOKEN=""

# Always asks when there is a terminal. The installed unit, then the environment,
# supply the default behind a blank answer, so re-running is two Enters and cannot
# blank a working token.
resolve_token() {
    local unit=$1 var=$2 current="" source="" hint
    if [[ -r "$SYSTEMD_DIR/$unit" ]]; then
        current="$(sed -n "s/^Environment=\"\?$var=\([^\"]*\)\"\?\$/\1/p" "$SYSTEMD_DIR/$unit" | tail -1)"
        [[ -z "$current" ]] || source="the installed unit"
    fi
    if [[ -z "$current" && -n "${!var:-}" ]]; then
        current="${!var}"
        source="\$$var in your environment"
    fi

    if [[ ! -t 0 ]]; then
        TOKEN="$current"
        [[ -z "$source" ]] || info "$var from $source"
    else
        if [[ -n "$current" ]]; then
            hint="blank keeps the ${#current}-character value from $source"
        else
            hint="blank leaves that server unauthenticated"
        fi
        read -rsp "  $var ($hint): " TOKEN < /dev/tty
        echo
        [[ -n "$TOKEN" ]] || TOKEN="$current"
    fi
    [[ "$TOKEN" != *[\"$'\n']* ]] || die "$var contains a quote or newline; systemd cannot carry that."
    [[ -n "$TOKEN" ]] || warn "$var empty - that server will accept unauthenticated requests"
}

install_unit() {
    local unit=$1 var=${2:-} tmp
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' RETURN
    if [[ -n "$var" ]]; then
        # Token via the environment, not -v: awk's argv is visible in ps.
        UNIT_TOKEN="$TOKEN" awk -v var="$var" '
            $0 ~ "^Environment=\"?" var "=" { printf "Environment=\"%s=%s\"\n", var, ENVIRON["UNIT_TOKEN"]; next }
            { print }' "$REPO/deploy/systemd/$unit" > "$tmp"
    else
        cat "$REPO/deploy/systemd/$unit" > "$tmp"
    fi
    sudo install -m 600 -o root -g root "$tmp" "$SYSTEMD_DIR/$unit"
    info "installed $unit"
}

SERVICES_INSTALLED=0
FRESH=0
for u in "${UNITS[@]}"; do [[ -e "$SYSTEMD_DIR/$u" ]] || FRESH=1; done

if [[ $CHECK_ONLY -eq 1 ]]; then
    for u in "${UNITS[@]}"; do
        [[ -e "$SYSTEMD_DIR/$u" ]] && info "$u installed" || warn "$u not installed"
    done
elif ! command -v systemctl >/dev/null 2>&1; then
    warn "no systemctl here; skipping"
else
    if [[ $FRESH -eq 0 ]]; then
        reply=y                       # refresh in place, tokens carried over
        info "already installed; refreshing from the repo"
    elif [[ -t 0 ]]; then
        read -rp "  Install the four units into $SYSTEMD_DIR (needs sudo)? [y/N] " reply < /dev/tty
    else
        reply=n
        warn "not a terminal; skipping (see the summary for the manual steps)"
    fi

    if [[ "$reply" =~ ^[Yy] ]]; then
        resolve_token mcp-science-server.service SCIENCEPCM_TOKEN
        install_unit  mcp-science-server.service SCIENCEPCM_TOKEN
        resolve_token mcp-openalex-server.service OPENALEX_TOKEN
        install_unit  mcp-openalex-server.service OPENALEX_TOKEN
        install_unit  mcp-prepare.service
        install_unit  mcp-tunnel.service
        TOKEN=""

        sudo systemctl daemon-reload
        if [[ $FRESH -eq 1 ]]; then
            sudo systemctl enable "${UNITS[@]}" >/dev/null
            info "enabled at boot, not started"
        fi
        SERVICES_INSTALLED=1
    fi
fi

# ---------------------------------------------------------------- summary

step "Ready"
cat <<EOF
  venvs   : $VENVS
  root    : $MCP_ROOT

  This script provisions the machine only. Each service pulls its own data, exports
  the shared reranker if it is missing, and builds its own index:

    source $MCP_ROOT/env.sh
    bash tools/sciencemcp-a100.sh prepare
    bash tools/openalex-a100.sh prepare

  Then run them:
    bash tools/sciencemcp-a100.sh serve
    bash tools/openalex-a100.sh serve

  Expose them (relay holds the public TLS endpoint):
    ./tools/mcp-tunnel.sh
EOF

if [[ $SERVICES_INSTALLED -eq 1 ]]; then
cat <<EOF

  The units are installed and enabled. Starting a server pulls in mcp-prepare, which
  runs both prepares in sequence first - hours on a wiped /datadisk:
    sudo systemctl start mcp-science-server mcp-openalex-server mcp-tunnel
    journalctl -u mcp-prepare -f

  Change a token later with:
    sudoedit $SYSTEMD_DIR/mcp-science-server.service && sudo systemctl daemon-reload
EOF
else
cat <<EOF

  Or install them as services (not mcp-console.service - that one runs on the relay):
    sudo cp deploy/systemd/mcp-{prepare,science-server,openalex-server,tunnel}.service \\
            $SYSTEMD_DIR/

  Set the tokens in the installed copies - the ones in the repo are empty on purpose,
  and an empty token starts the server unauthenticated:
    sudoedit $SYSTEMD_DIR/mcp-science-server.service    # SCIENCEPCM_TOKEN=
    sudoedit $SYSTEMD_DIR/mcp-openalex-server.service   # OPENALEX_TOKEN=

    sudo systemctl daemon-reload
    sudo systemctl enable --now mcp-prepare mcp-science-server mcp-openalex-server mcp-tunnel
EOF
fi
