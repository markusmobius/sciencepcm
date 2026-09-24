#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEMP_ROOT"' EXIT
export MCP_ROOT="$TEMP_ROOT/mcp"
export MCP_FAST_ROOT="$TEMP_ROOT/fast"
export CUSTOMMCP_CONFIG="$REPO/custom_mcps/pds.json"
export CUSTOMMCP_TOKEN="custommcp-test-only-token"
export COMMAND_LOG="$TEMP_ROOT/commands"
export CUSTOMMCP_TEST_EXIT=0
export CUSTOMMCP_PORT=8083
export PATH="$TEMP_ROOT/bin:$PATH"
OUTPUT="$TEMP_ROOT/output"
SCRIPT="$REPO/tools/custommcp-a100.sh"

mkdir -p "$TEMP_ROOT/bin" "$MCP_FAST_ROOT" "$MCP_ROOT/models/bge-reranker" "$MCP_ROOT/venvs/lab/bin"
touch "$MCP_ROOT/models/bge-reranker/model.onnx" "$MCP_ROOT/models/bge-reranker/tokenizer.onnx"
cat > "$TEMP_ROOT/bin/dotnet" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' 'DOTNET' "$@" >> "$COMMAND_LOG"
[[ "${CUSTOMMCP_TOKEN:-}" == custommcp-test-only-token ]] || exit 91
exit "$CUSTOMMCP_TEST_EXIT"
STUB
cat > "$MCP_ROOT/venvs/lab/bin/python" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' 'EXPORT' "$@" >> "$COMMAND_LOG"
touch "$MCP_ROOT/models/bge-reranker/model.onnx" "$MCP_ROOT/models/bge-reranker/tokenizer.onnx"
STUB
chmod +x "$TEMP_ROOT/bin/dotnet" "$MCP_ROOT/venvs/lab/bin/python"

assert_argument() {
    grep -Fxq -- "$1" "$COMMAND_LOG" || { echo "missing command argument: $1" >&2; exit 1; }
}

for script in "$SCRIPT" "$REPO/tools/gcr-prep.sh" "$REPO/tools/mcp-tunnel.sh"; do
    bash -n "$script"
done

bash "$SCRIPT" prepare --offline --collection neuralcomputation > "$OUTPUT" 2>&1
assert_argument "src/CustomMcp.Ingest"
assert_argument prepare
assert_argument "$CUSTOMMCP_CONFIG"
assert_argument "$MCP_ROOT/data/custommcp"
assert_argument "$MCP_FAST_ROOT/index/custommcp"
assert_argument --offline
assert_argument --collection
assert_argument neuralcomputation
if grep -Fxq EXPORT "$COMMAND_LOG"; then echo 'existing model was exported again' >&2; exit 1; fi

: > "$COMMAND_LOG"
bash "$SCRIPT" check > "$OUTPUT" 2>&1
assert_argument check
assert_argument "$CUSTOMMCP_CONFIG"

: > "$COMMAND_LOG"
bash "$SCRIPT" serve --rerank-candidates 17 > "$OUTPUT" 2>&1
assert_argument "$REPO/src/CustomMcp.Server"
assert_argument --gpu
assert_argument -p:UseGpu=true
assert_argument "$MCP_ROOT/models/bge-reranker"
assert_argument http://127.0.0.1:8083
assert_argument --rerank-candidates
assert_argument 17
if grep -Fq "$CUSTOMMCP_TOKEN" "$COMMAND_LOG" "$OUTPUT"; then echo 'token leaked into arguments or output' >&2; exit 1; fi

: > "$COMMAND_LOG"
CUSTOMMCP_PORT=8099 bash "$SCRIPT" > "$OUTPUT" 2>&1
assert_argument http://127.0.0.1:8099

: > "$COMMAND_LOG"
status=0
CUSTOMMCP_TEST_EXIT=17 bash "$SCRIPT" prepare > "$OUTPUT" 2>&1 || status=$?
[[ "$status" -eq 17 ]] || { echo 'prepare did not propagate the ingest failure' >&2; exit 1; }

: > "$COMMAND_LOG"
status=0
MCP_FAST_ROOT="$TEMP_ROOT/missing" bash "$SCRIPT" prepare > "$OUTPUT" 2>&1 || status=$?
[[ "$status" -ne 0 && ! -s "$COMMAND_LOG" ]] || { echo 'missing fast disk was not rejected before work' >&2; exit 1; }

: > "$COMMAND_LOG"
rm "$MCP_ROOT/models/bge-reranker/tokenizer.onnx"
bash "$SCRIPT" prepare --offline > "$OUTPUT" 2>&1
assert_argument EXPORT
assert_argument BAAI/bge-reranker-v2-m3
[[ -f "$MCP_ROOT/models/bge-reranker/tokenizer.onnx" ]]

grep -Fq 'mcp-custom-server.service' "$REPO/tools/gcr-prep.sh"
grep -Fq 'install_unit   mcp-custom-server.service CUSTOMMCP_TOKEN' "$REPO/tools/gcr-prep.sh"
grep -Fq 'bash tools/custommcp-a100.sh prepare' "$REPO/deploy/systemd/mcp-prepare.service"
grep -Fq '9204:8083' "$REPO/tools/mcp-tunnel.sh"
grep -Fq 'proxy_pass http://localhost:9204;' "$REPO/deploy/nginx/custommcp.econlabs.org.conf"

SYSTEMD_DIR="$TEMP_ROOT/units"
mkdir -p "$SYSTEMD_DIR"
printf '%s\n' 'Environment="CUSTOMMCP_TOKEN=retained-test-token"' > "$SYSTEMD_DIR/mcp-custom-server.service"
info() { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*"; }
die() { printf '%s\n' "$*" >&2; exit 1; }
sudo() { "$@"; }
source <(sed -n '/^resolve_secret() {/,/^}/p' "$REPO/tools/gcr-prep.sh")
resolve_secret mcp-custom-server.service CUSTOMMCP_TOKEN unauthenticated < /dev/null > "$OUTPUT" 2>&1
[[ "$SECRET" == retained-test-token ]] || { echo 'installed token was not preserved' >&2; exit 1; }
if grep -Fq "$SECRET" "$OUTPUT"; then echo 'installed token was printed' >&2; exit 1; fi
sudo() { return 19; }
status=0
( resolve_secret mcp-custom-server.service CUSTOMMCP_TOKEN unauthenticated < /dev/null ) > "$OUTPUT" 2>&1 || status=$?
[[ "$status" -ne 0 ]] || { echo 'unreadable installed credentials did not stop installation' >&2; exit 1; }
echo 'CustomMcp deployment contract checks passed.'