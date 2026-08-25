#!/usr/bin/env bash
# Reverse-tunnel the MCP server to a public host, so clients that cannot reach the
# GPU box directly connect to <relay>:9201 instead.
#
# The A100 sits behind the corporate network; the relay is reachable from anywhere.
# -R makes the relay listen and forward back down this connection, so nothing needs
# to be opened inbound on the GPU box.
#
# Run under screen/tmux, or install as a user systemd unit (see the bottom of this file).

set -uo pipefail

REMOTE="${REMOTE:-markusmobius@www.llmserver.econlabs.org}"
REMOTE_PORT="${REMOTE_PORT:-9201}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
RETRY_SECONDS="${RETRY_SECONDS:-5}"

if [[ ! -f "$KEY" ]]; then
  echo "No private key at $KEY. Set KEY=/path/to/key." >&2
  exit 1
fi

echo "tunnel: ${REMOTE}:${REMOTE_PORT} -> localhost:${LOCAL_PORT}"
echo "key   : ${KEY}"

while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting SSH tunnel ..."

  # ExitOnForwardFailure matters: without it a port already in use on the relay leaves
  # a connection that looks healthy but forwards nothing.
  ssh -i "$KEY" \
      -N \
      -o "BatchMode yes" \
      -o "ServerAliveInterval 30" \
      -o "ServerAliveCountMax 3" \
      -o "ExitOnForwardFailure yes" \
      -o "StrictHostKeyChecking accept-new" \
      -R "${REMOTE_PORT}:localhost:${LOCAL_PORT}" \
      "$REMOTE"

  echo "$(date '+%Y-%m-%d %H:%M:%S') connection closed (exit $?). Retrying in ${RETRY_SECONDS}s ..."
  sleep "$RETRY_SECONDS"
done

# Install as a user service instead of a screen session:
#
#   mkdir -p ~/.config/systemd/user
#   cat > ~/.config/systemd/user/mcp-tunnel.service <<'UNIT'
#   [Unit]
#   Description=SciencePCM MCP reverse tunnel
#   After=network-online.target
#
#   [Service]
#   ExecStart=%h/sciencepcm/tools/mcp-tunnel.sh
#   Restart=always
#   RestartSec=5
#
#   [Install]
#   WantedBy=default.target
#   UNIT
#   systemctl --user daemon-reload
#   systemctl --user enable --now mcp-tunnel
#   loginctl enable-linger "$USER"     # keeps it running when you log out
