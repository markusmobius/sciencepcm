# Operations

Day-to-day running of both services. Provisioning a fresh box is
[provisioning.md](provisioning.md); this assumes that is done.

## Tokens

Each service has its own bearer token, handed out independently. They live in the
installed unit files, not in the repo.

`tools/gcr-prep.sh` asks for both when it installs the units, and asks again on every
re-run so a token is never silently carried forward unseen. A blank answer keeps
whatever is already in the installed unit, or failing that the value of
`$SCIENCEPCM_TOKEN` / `$OPENALEX_TOKEN` in your shell — the prompt says which, and how
many characters it is. You can also `sudoedit` them afterwards:

```bash
sudoedit /etc/systemd/system/mcp-science-server.service     # Environment="SCIENCEPCM_TOKEN=..."
sudoedit /etc/systemd/system/mcp-openalex-server.service    # Environment="OPENALEX_TOKEN=..."
sudo systemctl daemon-reload
```

Use the values your existing clients already send. Generating fresh ones breaks every
configured client at once.

Left empty, a server starts unauthenticated and prints `auth: OPEN - no
SCIENCEPCM_TOKEN set`. Check that line after a restart — an empty value looks exactly
like a working one until someone reaches the endpoint without a token. `/health` never
needs a token, which is what separates "unreachable" from "wrong token" when debugging.

Unit files under `/etc/systemd/system` are world-readable by default; `gcr-prep.sh`
installs these `root:root` mode 600 instead. Running by hand rather than under systemd,
the servers read the same variables from the shell, or take `--token`.

## Running

```bash
source ~/mcp/env.sh
cd ~/sciencepcm

bash tools/sciencemcp-a100.sh prepare && bash tools/sciencemcp-a100.sh serve
bash tools/openalex-a100.sh   prepare && bash tools/openalex-a100.sh   serve
```

`prepare` pulls its service's data, exports the shared reranker if absent, and builds its
index only when `index-stamp.json` no longer matches the source shards and schema
version. Safe and cheap to re-run at any time. `check` reports paths, sizes and the index
schema version without starting anything.

Both scripts pass anything after `serve` to the server, e.g.
`serve --citation-prior 2.0`.

## As services

`tools/gcr-prep.sh` installs them. It asks once before touching `/etc`, prompts for each
token without echoing it, writes the units `root:root` mode 600, and enables them at boot
without starting them. Re-running refreshes the units from the repo and prompts again;
press Enter twice to keep the tokens that are already installed.

`mcp-console.service` is deliberately not installed here — it runs on the relay.

Start them when you are ready for the rebuild:

```bash
sudo systemctl start mcp-science-server mcp-openalex-server mcp-tunnel
```

| unit | job |
| --- | --- |
| `mcp-prepare` | runs both `prepare`s **in sequence**; the servers `Requires=` it |
| `mcp-science-server` | port 8080 |
| `mcp-openalex-server` | port 8081 |
| `mcp-tunnel` | reverse SSH, 9201→8080 and 9202→8081 |
| `mcp-console` | the browser console — runs on the *relay*, not here |

`mcp-prepare` exists because `/datadisk` is wiped on deallocation and both services then
rebuild from the same slow disk. Running them in sequence rather than in parallel is the
entire point; `TimeoutStartSec=0` because a rebuild takes hours. `systemctl status
mcp-prepare` sitting in `activating` with both servers queued behind it is the design
working, not a hang — watch it with `journalctl -u mcp-prepare -f`.

## Exposing them

The GPU box takes no inbound connections. A reverse SSH tunnel makes it appear on the
relay (`www.llmserver.econlabs.org`), where nginx terminates TLS.

```bash
./tools/mcp-tunnel.sh                          # both forwards
FORWARDS="9201:8080" ./tools/mcp-tunnel.sh     # just one
```

| relay port | local port | endpoint |
| --- | --- | --- |
| 9201 | 8080 | `https://www.sciencemcp.econlabs.org/mcp` |
| 9202 | 8081 | `https://www.openalexmcp.econlabs.org/mcp` |

`-R` binds to the relay's loopback, so the ports are never directly exposed — only nginx
reaches them. Verify with `ss -tlnp | grep 9201` on the relay.

The vhosts are in `deploy/nginx/`, each carrying its own install and certbot commands in
a header comment. **CORS lives in the app for servers we own and in nginx only for ones
we do not** — set in both places, browsers see a duplicated
`Access-Control-Allow-Origin` and reject the response outright.

## Checking the chain

In order, so a failure names its own hop:

```bash
curl -s localhost:8080/health                        # on the GPU box
curl -s localhost:9201/health                        # on the relay: tunnel up?
curl -s https://www.sciencemcp.econlabs.org/health   # anywhere: nginx and TLS
```

Each server loads its index and model at startup rather than on first request, so a wrong
path fails immediately instead of on someone's first question.

## Pointing an LLM at it

MCP over Streamable HTTP at `/mcp`. In VS Code, `.vscode/mcp.json`:

```json
{
  "servers": {
    "sciencepcm": {
      "type": "http",
      "url": "https://www.sciencemcp.econlabs.org/mcp",
      "headers": { "Authorization": "Bearer a-long-random-string" }
    }
  }
}
```

Clients should read parameter names from `tools/list` rather than hardcoding them. The
MCP SDK **silently drops unknown argument names**, so a client sending a misspelled or
outdated parameter gets the default back with no error — which is exactly how a customer
lost a day to `limit` once being called `k`.

For poking at either server by hand, `https://www.mcptest.econlabs.org` runs
`src/SciencePcm.Inspector` on the relay. Locally:

```bash
dotnet run --project src/SciencePcm.Inspector -c Release -- --urls http://localhost:6671
```

It discovers parameters from the schema, so new tool arguments appear without a code
change.

## Refreshing the data

Both corpora are produced elsewhere and pushed to the blob store; the A100 only pulls.

```powershell
# nerds21: re-ingest and upload
.\tools\nerds21-sync.ps1
.\tools\openalex-sync.ps1 -Force
```

```bash
# here: prepare notices the new digest and rebuilds
bash tools/sciencemcp-a100.sh prepare
bash tools/openalex-a100.sh prepare
```

No `--force` flag on this side. The blob store transfers only what differs, and the index
stamp decides whether a rebuild is needed.

## Disks

```
~/mcp/data/            pulled from the cloud, disposable, ~145 GB
/datadisk/index/       built here and only here, ~765 GB
```

`/datadisk` is local NVMe (1.4 GB/s) and ephemeral. The OpenAlex index is larger than the
free space on the 1 TB OS disk, so **there is no durable copy** — a deallocation costs a
full rebuild rather than an rsync. Both scripts fail rather than fall back to the OS disk
if `/datadisk` is missing.

Storage was the whole latency story once: the index on a managed disk gave 95% iowait and
6s queries. Moving it to NVMe took a short query to 1.24s, and parallel segment search
took a long one from 5.66s to 0.44s.

## Before the reservation lapses

The A100 is a working copy; nerds21 is the durable archive. Anything not in the blob
store is lost when the reservation ends. Indexes are rebuildable and need not be kept —
the corpora and digests are what matter, and they are already in the store.
