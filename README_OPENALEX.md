# OpenAlex abstract MCP

This is the second MCP service. It shares the tested Lucene and cross-encoder engine
with SciencePCM, but every executable, artifact path, cloud path, token, tool name and
network endpoint is OpenAlex-specific.

The corpus is every work in the local OpenAlex snapshot for which
`abstract_inverted_index` is nonempty. There is no neuroscience, year, topic or field
filter and no full-text tier.

The v2 digest is designed to resolve paper mentions in newspaper articles. In addition
to title and abstract it retains publication date/year, authors, institutions, journal,
ISSN, DOI, PMID, language, work type, citation count, volume/issue/pages, topics,
keywords and retraction status. Lucene searches title, abstract, authors, institutions,
journal, DOI, ISSN, topics and keywords; the other fields are returned for verification.

## Programs

| Program | Runs on | Purpose |
| --- | --- | --- |
| `OpenAlex.Ingest` | nerds21 | Reconstruct abstracts and write Parquet shards. |
| `openalex-sync.ps1` | nerds21 | Run ingest and upload `openalex/abstracts`. |
| `OpenAlex.Index` | A100 | Build the stored-field Lucene index. |
| `openalex-a100.sh` | A100 | Pull, prepare, index and serve on port 8081. |
| `OpenAlex.Server` | A100 | Serve the OpenAlex MCP endpoint. |

## 1. nerds21

The share `\\nerds21\OpenAlexData` is `C:\OpenAlexData` locally on nerds21.
Use the local path on nerds21; SMB loopback only makes a long ingest slower.

First inspect the plan:

```powershell
cd D:\SciencePCM\mcpserver\repo
.\tools\openalex-sync.ps1 -CheckOnly
```

Then run the ingest and upload. This is the long corpus job:

```powershell
.\tools\openalex-sync.ps1
```

Output lands at `C:\OpenAlexData\__temp\abstracts`, visible as
`\\nerds21\OpenAlexData\__temp\abstracts`. It is uploaded to
`openalex/abstracts` with the cloud tag `project=openalex`.

After replacing or updating the snapshot, rebuild and overwrite the same cloud path:

```powershell
.\tools\openalex-sync.ps1 -Force
```

Use `-Force` after changing the digest schema as well. The sync check compares file
names, and rebuilt shards deliberately reuse the same names.

For a bounded ingest smoke test without cloud transfer:

```powershell
dotnet run --project src\OpenAlex.Ingest -c Release -- `
  --input C:\OpenAlexData\data\works `
  --out C:\OpenAlexData\__temp\smoke `
  --limit 10000 --threads 8 --shard-size 5000
```

## 2. A100

This assumes the machine has already been provisioned by `tools/gcr-prep.sh`, so its
.NET, sync/lab Python environments and CUDA 12 libraries are available.

```bash
cd ~/sciencepcm
source ~/sciencepcm-data/env.sh
bash tools/openalex-a100.sh check
bash tools/openalex-a100.sh prepare
```

`prepare` pulls the digest into `~/openalex-data/abstracts`, exports the general-domain
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, builds with GPU ONNX Runtime, and
creates `~/openalex-data/index/abstracts-bm25`. It skips completed stages on rerun.

The corpus is multilingual, while Lucene's analyzer and this reranker are English
oriented. Non-English records are retained but retrieval quality will be weaker. A
truly multilingual reranker requires extending the C# tokenizer beyond BERT WordPiece;
it should not be silently substituted without parity tests.

Set an independent token and start the second server:

```bash
export OPENALEX_TOKEN='a-different-long-random-string'
echo "export OPENALEX_TOKEN='$OPENALEX_TOKEN'" >> ~/.bashrc

screen -S openalex-mcp
source ~/sciencepcm-data/env.sh
cd ~/sciencepcm
bash tools/openalex-a100.sh serve
```

Detach with `Ctrl+A`, `D`, then verify:

```bash
curl -s localhost:8081/health
```

The response identifies `service: OpenAlex` and reports the abstract count.

## 3. Tunnel and nginx

Run both reverse forwards from the A100 machine:

```bash
FORWARDS="9201:8080 9202:8081" ./tools/mcp-tunnel.sh
```

On the relay, install `deploy/nginx/openalexmcp.econlabs.org.conf` and obtain its TLS
certificate using the commands in that file. The endpoint is then:

```text
https://www.openalexmcp.econlabs.org/mcp
```

The MCP tools are deliberately distinct:

- `search_openalex`
- `get_openalex_work`
- `openalex_corpus_stats`