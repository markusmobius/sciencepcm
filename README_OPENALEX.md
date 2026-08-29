# OpenAlex abstract MCP

This is the second MCP service. It shares the tested Lucene and cross-encoder engine
with SciencePCM, but every executable, artifact path, cloud path, token, tool name and
network endpoint is OpenAlex-specific.

The corpus is every work in the local OpenAlex snapshot that has an abstract. There is
no neuroscience, year, topic or field filter and no full-text tier.

The v3 ingest keeps works that have only a title, and the index build discards them
again with `--require-body`. That looks redundant and is deliberate: the Parquet digest
stays a superset so the decision can be revisited without a four-hour re-ingest.

Indexing title-only works was a mistake, and the reasoning behind it was wrong twice
over. It was introduced because landmark papers — the ChAdOx1 Lancet trial, Jinek 2012 —
were missing from search results, which was attributed to publishers not depositing
abstracts to OpenAlex. Both papers were in fact present, with abstracts, all along. The
change tripled the corpus to 485M documents and dropped the average field length to ~97
tokens, which made BM25's length penalty roughly three times harsher for every real
paper. See "Why landmark papers were missing" below.

## Ranking

Three defaults keep the paper itself above the literature about it:

- `--exclude-types peer-review,dataset,paratext` drops work types that OpenAlex records
  as separate works titled after the paper they discuss. Pass an empty value to disable.
- `--citation-prior 1.0` multiplies the BM25 score by up to `1 + weight`, scaled by
  `log10(1 + cited_by_count)`. Multiplicative and capped rather than additive: an
  additive bonus has to be tuned against BM25's score scale, and at any weight large
  enough to matter the most-cited papers in the corpus win every query outright. Pass
  `0` to disable. Measured on the landmark set: MRR 0.125 → 0.333.
- Results are deduplicated by title to the **best-cited** record, not the first one seen.
  OpenAlex holds the same paper more than once — a preprint, a stub typed `other`, a
  merged-but-not-removed record — and the copies carry few or no citations.

`--max-doc-freq-ratio` and `--bm25-b` exist and are off by default. Both were tried
against the landmark set and neither improved it; they are kept as diagnostics.

## Why landmark papers were missing

Four papers known to be in OpenAlex were absent from the top 50 of queries that named
them. `eval/known_item.py` scores this without an LLM, and `OpenAlex.Index explain`
prints Lucene's own account of a score. The explanation for the ChAdOx1 paper:

```
fieldLength    = 2621.44
avgFieldLength =   96.71
chadox1: idf=11.74 (docFreq 3,874)  x  tfNorm=0.0856  =  1.005
total = 14.41
```

The rarest, most decisive term in the query contributed 1.005 out of 14.41, because the
document is 27x the average length and BM25 divides by that. It was long because
`SearchableText` concatenated all 765 author names and 64 institutions — 15,000
characters, about 80% of the document. Papers by large collaborations were therefore
buried under the news pieces, errata and commentaries written about them, while papers
with short author lists ranked first.

Author and institution lists are now capped at 400 characters each in the indexed text,
which takes that document from ~2,621 tokens to ~300. News prose names lead authors, so
the cap keeps the useful part.



The v3 digest is designed to resolve paper mentions in newspaper articles. In addition
to title and abstract it retains publication date/year, authors, institutions, journal,
ISSN, DOI, PMID, language, work type, citation count, volume/issue/pages, topics,
keywords and retraction status. Lucene searches title, abstract, authors, institutions,
journal, DOI, ISSN, topics and keywords; the other fields are returned for verification.

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

## Upgrading a v2 deployment to v3

The schema, the corpus filter and the reranker all changed, so nothing is reusable.
The v3 digest holds 484,677,603 works against v2's 256,108,425, but the index build
passes `--require-body` and keeps only the ~250M with abstracts.

1. On nerds21, re-ingest and overwrite the cloud path: `.\tools\openalex-sync.ps1 -Force`.
2. On the A100, stop the server, then clear both disks at once:

   ```bash
   bash tools/openalex-a100.sh clean          # lists what it would remove
   bash tools/openalex-a100.sh clean --yes    # actually removes it
   ```

   That drops the digest, the index on *both* the managed disk and `/datadisk`, the
   retired MiniLM reranker and any half-finished pull.
3. `bash tools/openalex-a100.sh prepare`, then `serve` with no extra arguments. The
   defaults are the settled configuration; every flag that was tried and rejected is
   off unless passed explicitly.

## Disk layout

`prepare` splits the two artifacts across the two disks deliberately:

| artifact | location | why |
| --- | --- | --- |
| Parquet digest (134 GB) | `~/openalex-data/abstracts` (managed disk) | read once, sequentially, during the build |
| Lucene index (~400-450 GB) | `/datadisk/openalex-data/index` (NVMe) | random reads and segment merges need the IOPS |
| durable index copy | `~/openalex-data/index` (managed disk) | `/datadisk` is wiped on deallocate |

The index is built on NVMe and then rsynced to the durable copy, so both locations end
up populated in one pass. `datadisk-restore.service` syncs the durable copy back after a
deallocation, which costs minutes against hours to rebuild.

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