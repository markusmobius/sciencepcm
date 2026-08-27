# OpenAlex abstract MCP

This is the second MCP service. It shares the tested Lucene and cross-encoder engine
with SciencePCM, but every executable, artifact path, cloud path, token, tool name and
network endpoint is OpenAlex-specific.

The corpus is every work in the local OpenAlex snapshot that has a title or an abstract.
There is no neuroscience, year, topic or field filter and no full-text tier.

Requiring a nonempty `abstract_inverted_index`, as v2 did, was a mistake. Elsevier and
AAAS supply no abstracts to OpenAlex, so the ChAdOx1 Lancet paper and Jinek 2012 were
both absent from the index while the peer reviews and Faculty Opinions records *about*
them, which carry their titles, were present and ranked highly. A work with only a
title is still matchable, because news prose names a paper by its title, authors,
institutions and journal far more often than it quotes the abstract.

The v3 digest is designed to resolve paper mentions in newspaper articles. In addition
to title and abstract it retains publication date/year, authors, institutions, journal,
ISSN, DOI, PMID, language, work type, citation count, volume/issue/pages, topics,
keywords and retraction status. Lucene searches title, abstract, authors, institutions,
journal, DOI, ISSN, topics and keywords; the other fields are returned for verification.

## Ranking

Two defaults exist to keep the paper itself above the literature about it:

- `--exclude-types peer-review,dataset,paratext` drops work types that OpenAlex records
  as separate works titled after the paper they discuss. Pass an empty value to disable.
- `--citation-prior 0.5` adds `0.5 * log10(1 + cited_by_count)` to the rerank score.
  It only ever adds, because OpenAlex sometimes holds duplicate records of one paper
  with the citations split or lost between them; a zero-citation record keeps its rerank
  score rather than being pushed down. Pass `0` to disable.

The reranker is `BAAI/bge-reranker-v2-m3`. On 30 known-item news-to-paper queries it
reached hit@1 40.0% and hit@10 76.7%, against 23.3% and 70.0% for
`cross-encoder/ms-marco-MiniLM-L-6-v2`, for about 475 ms more per query. It is XLM-R
based, so unlike the previous reranker it matches the multilingual corpus. Raising
`--rerank-candidates` from 100 to 500 changed neither hit@1 nor hit@10, so the default
stays at 100.

## Programs

| Program | Runs on | Purpose |
| --- | --- | --- |
| `OpenAlex.Ingest` | nerds21 | Reconstruct abstracts and write Parquet shards. |
| `openalex-sync.ps1` | nerds21 | Run ingest and upload `openalex/abstracts`. |
| `OpenAlex.Index` | A100 | Build the stored-field Lucene index. |
| `openalex-a100.sh` | A100 | Pull, prepare, index and serve on port 8081. |
| `OpenAlex.Server` | A100 | Serve the OpenAlex MCP endpoint. |

## Upgrading a v2 deployment to v3

The schema, the corpus filter and the reranker all changed, so nothing is reusable.
Run the full sequence:

1. On nerds21, re-ingest and overwrite the cloud path: `.\tools\openalex-sync.ps1 -Force`.
2. On the A100, discard the old digest, index and reranker so `prepare` rebuilds them:
   `rm -rf ~/openalex-data/abstracts ~/openalex-data/index ~/openalex-data/models/openalex-cross`.
3. `bash tools/openalex-a100.sh prepare`, then restart the server.

The index grows: v3 keeps works v2 dropped, and `work_type` is now an indexed term
rather than a stored-only field.

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

`prepare` pulls the digest into `~/openalex-data/abstracts`, exports the
`BAAI/bge-reranker-v2-m3` reranker, builds with GPU ONNX Runtime, and creates
`~/openalex-data/index/abstracts-bm25`. It skips completed stages on rerun.

The corpus is multilingual and so is the reranker, but Lucene's analyzer is English
oriented, so non-English records are retained with weaker recall.

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