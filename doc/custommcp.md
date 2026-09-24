# CustomMcp

CustomMcp hosts named paper collections from cloud PDS files. It replaces the
undeployed MIT-branded PDS server, not the existing MITMCP JATS archive. ScienceMCP,
OpenAlex MCP, and MITMCP keep their endpoints, data, and credentials.

| Service | A100 port | Relay port | Unit | Public route |
| --- | --- | --- | --- | --- |
| CustomMcp | 8083 | 9204 | `mcp-custom-server` | `www.custommcp.econlabs.org/mcp/<name>` |

The A100 wrapper follows the same `check`, `prepare`, and `serve` convention as the
other services. It reuses `~/mcp/models/bge-reranker`, so an existing A100 does not
need another model export. All CustomMcp collections share one loaded ONNX session
and rerank limit; each has separate paper and passage Lucene readers.

## Configure Collections

[custom_mcps/pds.json](../custom_mcps/pds.json) is the default configuration for both
preparation and serving:

```json
{
  "customCorpora": [
    {
      "name": "neuralcomputation",
      "inputCloudPath": "sciencepcm/Sept222026/input_2026_09_23_fixed.pds"
    }
  ]
}
```

The entry creates `/mcp/neuralcomputation`. Names must be unique lowercase slugs,
using letters, digits, hyphens or underscores, and beginning with a letter or digit.
They also namespace article keys: `neuralcomputation:<PDS key>`. Renaming a collection
changes its endpoint and keys; prepare and restart after a configuration change.

Add another entry for another PDS. No new service, tunnel, or nginx location is
required. The generic nginx prefix preserves `/mcp/<name>`. There is deliberately
no aggregate `/mcp` route or fallback for an unknown name. All endpoints use the
same `CUSTOMMCP_TOKEN`; collection scoping is not per-collection authorization.

This configuration is **not** [custom_mcps/config.json](../custom_mcps/config.json).
That file belongs to OpenAlex and defines ID allowlists over one existing OpenAlex
index. PDS collections each need their own ingestion and two indexes.

PDS is a container, not a universal paper schema. The initial supported records have
`metadata` (title, ordered authors, journal, optional DOI and `publish date`),
`abstract` sentence objects or text, and nested `body.sections` with `header title`,
`text`, `subsections`, and optional table captions. Unsupported record shapes fail
preparation; they are not silently skipped. Known Neural Computation checkpoint
filenames can supply a missing DOI, but arbitrary PDF filenames cannot.

## Deploy on the Existing A100

Use an updated checkout on the A100. Run provisioning interactively in your own
terminal so secrets stay out of chat and the repository:

```bash
cd ~/sciencepcm
bash tools/gcr-prep.sh
source ~/mcp/env.sh
bash tools/custommcp-a100.sh prepare
bash tools/custommcp-a100.sh check
sudo systemctl start mcp-custom-server
sudo systemctl restart mcp-tunnel
curl -fsS http://127.0.0.1:8083/health
```

The installer now includes `mcp-custom-server`, prompts for `CUSTOMMCP_TOKEN`, retains
existing credentials when you accept their defaults, and rewrites account paths.
Set `legopds_clienthash` for preparation; it is independent of the serving token.
The service reads its token from the installed unit, not process arguments. Empty
serving tokens mean unauthenticated access, so set one before exposing the endpoint.

On an already-running box, manually preparing CustomMcp is important:
`mcp-prepare` may already be active with `RemainAfterExit=yes`, so starting the new
server does not rerun preparation. You do not need to restart the three existing
server processes or their preparation unit. Restarting the shared tunnel briefly
reconnects the existing forwards and adds `9204:8083`.

On future boots, the shared `mcp-prepare` unit runs all four preparations in sequence.
A failed preparation prevents the dependent servers from starting; CustomMcp does
not silently accept cached data after a failed cloud refresh.

## Expose on the Relay

Point DNS for `www.custommcp.econlabs.org` at the existing relay. Confirm port 80 is
reachable for certificate issuance. With an updated checkout on the relay, obtain
the certificate **before** enabling the supplied TLS-only vhost:

```bash
sudo certbot certonly --nginx -d www.custommcp.econlabs.org
sudo install -m 644 deploy/nginx/custommcp.econlabs.org.conf /etc/nginx/sites-available/custommcp
sudo ln -sfn /etc/nginx/sites-available/custommcp /etc/nginx/sites-enabled/custommcp
sudo nginx -t
sudo systemctl reload nginx
curl -fsS http://127.0.0.1:9204/health
curl -fsS https://www.custommcp.econlabs.org/health
```

The configured collection's MCP URL is
`https://www.custommcp.econlabs.org/mcp/neuralcomputation`.
Use `Authorization: Bearer <CUSTOMMCP_TOKEN>` on MCP requests. `/health` is
unauthenticated and lists each loaded collection and its counts. CORS is set by the
application only; do not add another set of CORS headers in nginx.

The inspector's default configuration includes this endpoint. Redeploy/restart the
console on the relay to load that configuration. Further collections can be added
to the inspector's server list independently of configuring the MCP host.

## Prepare and Refresh

```bash
source ~/mcp/env.sh
bash tools/custommcp-a100.sh prepare
sudo systemctl restart mcp-custom-server
```

Preparation refreshes each configured cloud PDS through the published
BlobDataMachine NuGet package. There is no dependency on the producer repository,
and no separate Windows Parquet upload step for CustomMcp.

The source SHA256, collection name/path, ingest version, and chunk settings identify
the Parquet revision. The index schema also identifies the index generation. Both
indexes join article metadata, preserving authors and title-only records. Existing
unchanged shards and indexes are reused. A new source builds separately, then
atomically updates a small readiness marker only after both index counts validate.
Failure leaves the last completed marker intact. The running server holds its
existing readers until restarted; it refuses a marker bound to a different config.

```text
~/mcp/data/custommcp/
  cache/<cloud path>                  downloaded PDS files
  <name>/<revision>/abstracts/        paper Parquet
  <name>/<revision>/passages/         article metadata and passage Parquet
  <name>/<revision>/ingest-report.json
/datadisk/index/custommcp/
  <name>/current.json                 completed generation and source identity
  <name>/<generation>/abstracts/      Lucene paper index
  <name>/<generation>/passages/       Lucene passage index
```

Old revisions are retained, not deleted automatically. Remove superseded revisions
only after the serving process has switched away from them. If `/datadisk` is wiped,
prepare rebuilds the indexes from the surviving shards. No service silently moves
its indexes to the OS disk when the fast disk is unavailable.

For an explicitly offline recovery with existing cached PDS files:

```bash
bash tools/custommcp-a100.sh prepare --offline
```

This cannot detect upstream source changes. Running it manually does not satisfy a
failed `mcp-prepare` systemd dependency: restore cloud access before starting the
dependent units, or run CustomMcp manually with `serve` after the offline preparation.
Network-free serving also requires the shared model to be present already.

`--collection neuralcomputation` limits
preparation to one configured entry. `--input /path/to/input.pds` uses a local PDS
and requires exactly one selected collection. Use `--help` on `CustomMcp.Ingest`
for chunk sizes, shards, and other options.

The wrapper accepts `MCP_ROOT`, `MCP_FAST_ROOT`, `CUSTOMMCP_CONFIG`, and
`CUSTOMMCP_PORT`. A config override must be the same during prepare and serve;
under systemd, set it for both the preparation and server units. Port changes also
require a matching tunnel forward. The default model directory is shared with the
other services.

## Local Development

Run from the repository root. This builds both indexes without requiring a model:

```powershell
dotnet run --project src/CustomMcp.Ingest -c Debug -- prepare `
  --config custom_mcps/pds.json --data-root runs/custommcp/data `
  --index-root runs/custommcp/index
```

To serve locally, supply an actual exported model directory:

```powershell
dotnet run --project src/CustomMcp.Server -c Debug -- `
  --config custom_mcps/pds.json --index-root runs/custommcp/index `
  --cross-encoder C:/path/to/bge-reranker --urls http://127.0.0.1:8083
```

The server project copies the default PDS config into build/publish output and needs
only that configuration, completed indexes, and model. It has no cloud credential or
PDS-package dependency. Use `dotnet publish` for a standalone application directory.
GPU builds use `-p:UseGpu=true` plus runtime `--gpu`; the A100 wrapper supplies both.

## Included Collection and Verification

The repaired Neural Computation PDS contains **364 papers, 363 published abstracts,
and 14,960 passages** with default chunking. The erratum without an abstract remains
retrievable by title, author, and DOI. Three upstream failed or missing papers are
not present. The source is 8,820,737 bytes with SHA256
`a90d6ce0278330e4d837c3bd70817cd89c88ea4ff8426e99395103deec818707`.

This snapshot has no publication dates: they remain unknown, displayed as year zero,
and either year bound excludes them. Citation counts and access status are not
inferred. Mathematical text may still contain extraction artifacts. The existing
MIT-specific review/repair commands remain in `MitPcm.Ingester`; routine CustomMcp
preparation does not run them or modify a cloud PDS.

```bash
dotnet test SciencePcm.slnx -c Debug
bash tests/deploy/custommcp.sh
```

Tests cover ingestion, stable namespaced keys, protected preparation, and all five
MCP tools over multiple independent collections, including concurrent requests,
cross-collection lookups, auth, and CORS. A tiny deterministic ONNX fixture verifies
inference plumbing, not relevance quality or GPU performance. Shell tests stub
external commands and do not install units, change live services, or export models.