# OpenAlex abstract MCP

This is the second MCP service. It shares the tested Lucene and cross-encoder engine
with SciencePCM, but every executable, artifact path, cloud path, token, tool name and
network endpoint is OpenAlex-specific.

The corpus is every work in the local OpenAlex snapshot that has a title or an abstract:
484,677,603 works, of which 234M have no abstract. There is no neuroscience, year, topic
or field filter and no full-text tier.

Abstract-less works are kept because they are mostly real papers, not noise. Measured
against the OpenAlex API: 21.6% of 2025 articles have no abstract, rising to **29.1% of
2025 articles with 10 or more citations**, and 44.9% across all articles. The fielded
index handles them without a special case, so no switch is needed.

`OpenAlex.Index build --require-body` excludes them, for measuring the corpus both ways.
It is not the default.

Note for anyone re-reading the history here: an earlier version of this file blamed
missing landmark papers on publishers not depositing abstracts. That was wrong — the
papers had abstracts throughout. See "Why landmark papers were missing".

## Fields and ranking

Fielded BM25F then a cross-encoder, shared with ScienceMCP — see
[retrieval.md](retrieval.md) for the field boosts, the citation prior, deduplication and
the approaches that were tried and rejected.

Two defaults are specific to this service:

- `--exclude-types peer-review,dataset,paratext` drops work types OpenAlex records as
  separate works titled after the paper they discuss.
- `--citation-prior 1.0`, which matters more here than for ScienceMCP because the corpus
  holds the errata, commentaries and duplicate records that quote a paper's title.

## Filters

`search_openalex` takes `author`, `journal` and `sort` alongside `query`:

- `author` matches the OpenAlex form exactly — `"Duflo, Esther"` — against one indexed
  term per name, so it lists a researcher's papers rather than matching two common words.
- `journal` matches an exact venue name.
- `sort` is `relevance` (default), `citations` or `year`.
- `query` is optional when `author` or `journal` is set, which is how you browse.

Reranking is skipped for a non-relevance sort: a browse has no relevance signal to rerank
on, and reordering would defeat the sort that was asked for.

## Why landmark papers were missing

Worth reading before touching ranking: four papers known to be in OpenAlex were absent
from the top 50 of queries that named them, and the cause was neither the corpus nor the
reranker. The investigation, the Lucene `explain` output and the false trail that cost a
day are in [retrieval.md](retrieval.md#diagnosing-a-missing-paper).

The short version: `SearchableText` concatenated all 765 author names into one searched
field, making the ChAdOx1 paper 27x the average document length, so BM25 scaled its
decisive `chadox1` term — IDF 11.74 — down to 1.005 out of a total 14.41. The fielded
schema fixed it by construction.

The digest is designed to resolve paper mentions in newspaper articles. In addition
to title and abstract it retains publication date/year, authors, institutions, journal,
ISSN, DOI, PMID, language, work type, citation count, volume/issue/pages, topics,
keywords and retraction status.

## Why there is no Semantic Scholar merge

Measured with `eval/s2_probe.py` on 500 works matching
`has_abstract:false,type:article,cited_by_count:>50` — the exact population a merge
would target. 92.6% carry a DOI and 84.0% of those resolve in S2, so the join works;
titles agree 92.5% of the time, meaning roughly 1 in 13 DOIs points at a different
paper in the two databases and any merge would have to verify the title before
accepting anything.

S2 supplies an abstract for only **16.0%** of them. On a control cohort where OpenAlex
already has an abstract the rate is 29.2%, so S2's coverage is limited by
redistribution rights rather than by whether the abstract exists: it fails on the same
paywalled publishers OpenAlex fails on. A merge would recover a sixth of the gap for a
bulk download, a DOI join and a re-ingest.

33.3% of the same cohort do have an open-access PDF, which is where S2ORC full text
would come from. That remains the only reason to revisit Semantic Scholar, and it is a
full-text project rather than an abstract one.

## Reranker

The reranker is `BAAI/bge-reranker-v2-m3`. On 30 known-item news-to-paper queries it
reached hit@1 40.0% and hit@10 76.7%, against 23.3% and 70.0% for
`cross-encoder/ms-marco-MiniLM-L-6-v2`, for about 475 ms more per query. It is XLM-R
based, so unlike the previous reranker it matches the multilingual corpus.

`--rerank-candidates` stays at 100. Raising it to 500 changed nothing, and 1,000 made
the landmark set slightly worse — a deeper pool is more noise for the cross-encoder to
sift. The cross-encoder is not the weak stage: on the landmark set it lifted papers from
BM25 rank 14 to rank 1, and rescued papers BM25 ranked outside its own top 50. When a
paper is missing from results, look at the first stage.

## Programs

| Program | Runs on | Purpose |
| --- | --- | --- |
| `OpenAlex.Ingest` | nerds21 | Reconstruct abstracts and write Parquet shards. |
| `openalex-sync.ps1` | nerds21 | Run ingest and upload `openalex/abstracts`. |
| `OpenAlex.Index` | A100 | Build the stored-field Lucene index. |
| `openalex-a100.sh` | A100 | Pull, prepare, index and serve on port 8081. |
| `OpenAlex.Server` | A100 | Serve the OpenAlex MCP endpoint. |

## Upgrading, or just re-running

There is one command, and it is safe to run at any time:

```bash
bash tools/openalex-a100.sh prepare
```

It is a waterfall, and each stage decides for itself whether there is work to do:

1. **Digest** — pulled every time. The blob store fingerprints files and transfers only
   what differs, so an unchanged digest costs nothing and a rebuilt one is picked up
   automatically.
2. **Reranker** — exported only when the ONNX files are absent. Both services share one
   `bge-reranker-v2-m3` export; the weights are identical.
3. **Index** — the builder writes `index-stamp.json` beside the index recording the
   schema version and a fingerprint of the source shards. If both still match it returns
   immediately; if the digest changed or the field layout changed, it rebuilds. Bump
   `LexicalIndex.SchemaVersion` when the fields change and every index invalidates itself.

So a schema change or a new digest needs no manual cleanup: pull the code and run
`prepare`. There is no `clean` command, because deciding what is stale is the tool's job.

## Disk layout

One root for everything downloaded, one directory for everything built:

```
~/mcp/
   env.sh
   venvs/     sync, eval, lab, cuda12
   models/    bge-reranker  (shared by both services)
   data/      sciencepcm/{abstracts,passages-2019-2025,questions}
              openalex/abstracts            134 GB
/datadisk/index/
   science-abstracts/  science-passages/  openalex-abstracts/
```

Everything under `data/` came from the cloud and can be deleted and re-pulled; the
per-service subdirectory is the cloud path, which is why the pull lands in place.

| artifact | location | why |
| --- | --- | --- |
| Parquet digest (134 GB) | `~/mcp/data/openalex/abstracts` (managed disk) | read once, sequentially, during the build |
| Lucene index (~745 GB) | `/datadisk/index/openalex-abstracts` (NVMe) | random reads and segment merges need the IOPS |

**There is no durable copy of the OpenAlex index.** It is larger than the free space on
the 1 TB OS disk, so `/datadisk` is its only home and a deallocation costs a rebuild from
Parquet rather than an rsync. `prepare` notices on its own, because the stamp is wiped
along with the disk. Both scripts fail rather than fall back to the OS disk if
`/datadisk` is missing.

Both bottlenecks are real and independent: moving the index to NVMe took a short query
from 6.65s to 1.24s, while a long natural-language query was unaffected at 5.66s until
parallel segment search took it to 0.44s. Neither fix substitutes for the other, and the
NVMe argument strengthens as the index grows past the box's 216 GB of RAM.

Set `OPENALEX_FAST_ROOT` to override `/datadisk`. If it is missing or not writable the
script serves from the managed disk and says so.

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
source ~/mcp/env.sh
bash tools/openalex-a100.sh check
bash tools/openalex-a100.sh prepare
```

`prepare` pulls the digest into `~/mcp/data/openalex/abstracts`, exports the
`BAAI/bge-reranker-v2-m3` reranker if it is not already there, and builds
`/datadisk/index/openalex-abstracts`. It skips whatever is already current.

The corpus is multilingual and so is the reranker, but Lucene's analyzer is English
oriented, so non-English records are retained with weaker recall.

Set an independent token and start the second server:

```bash
export OPENALEX_TOKEN='a-different-long-random-string'
echo "export OPENALEX_TOKEN='$OPENALEX_TOKEN'" >> ~/.bashrc

screen -S openalex-mcp
source ~/mcp/env.sh
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