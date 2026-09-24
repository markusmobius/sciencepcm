# SciencePCM

Scientific literature MCP services using BM25 retrieval followed by an ONNX
cross-encoder rerank. The deployment topology has four server processes:

| Service | Corpus | A100 / relay ports | Public MCP route |
| --- | --- | --- | --- |
| ScienceMCP | 5.3M neuroscience abstracts + 15M full-text passages | 8080 / 9201 | `www.sciencemcp.econlabs.org/mcp` |
| OpenAlex MCP | 485M OpenAlex works and configured ID subsets | 8081 / 9202 | `www.openalexmcp.econlabs.org/mcp` and `/mcp/<name>` |
| MITMCP | Existing 10-journal MIT Press JATS archive, 1989-2025 | 8082 / 9203 | `www.mitmcp.econlabs.org/mcp` |
| CustomMcp | Named cloud PDS collections | 8083 / 9204 | `www.custommcp.econlabs.org/mcp/<name>` |

The first three services already exist. CustomMcp is the new fourth service;
deployment files are included, but the A100 unit and relay DNS/TLS must be installed
as described in [doc/custommcp.md](doc/custommcp.md#deploy-on-the-existing-a100).

MITMCP is a separate instance of `SciencePcm.Server`, not a route inside ScienceMCP.
[tools/mit-sync.ps1](tools/mit-sync.ps1) ingests its JATS archive and uploads Parquet
under `mitmcp/abstracts` and `mitmcp/passages`;
[tools/mitmcp-a100.sh](tools/mitmcp-a100.sh) pulls and indexes that data.
The four services share retrieval code and exported model files, but each process
loads its own inference session. Within CustomMcp, all named collections share one
session and rerank concurrency limit, with separate paper and passage indexes.

A browser console for the services lives at `www.mcptest.econlabs.org`.

OpenAlex also serves ID-restricted corpora from the same process and port, configured
in [custom_mcps/config.json](custom_mcps/config.json). The included MIT subset is at
`/mcp/openalexmit`; the full corpus stays at `/mcp`. See
[custom corpus endpoints](doc/openalex.md#custom-corpus-endpoints) for configuration.

**CustomMcp** is configured in [custom_mcps/pds.json](custom_mcps/pds.json): each entry
has a `name` and `inputCloudPath`. The included `neuralcomputation` collection points
to the repaired 364-paper PDS, with 363 abstracts and 14,960 passages, and is served
at `/mcp/neuralcomputation`. Adding another supported PDS needs a config entry,
`prepare`, and a restart, not another server project or systemd unit. This does not
replace MITMCP or the OpenAlex MIT subset. There is no unscoped CustomMcp `/mcp` route.

## Documentation

- **[doc/provisioning.md](doc/provisioning.md)** — getting a GPU box ready: the right
  machine, .NET, Python environments, the CUDA 12 / cuDNN 9 dance ONNX Runtime needs.
- **[doc/operations.md](doc/operations.md)** — running it: tokens, the systemd units, the
  reverse tunnel and nginx, refreshing the data, what lives on which disk.
- **[doc/sciencemcp.md](doc/sciencemcp.md)** — the neuroscience service: two tiers, its
  tools, and why full text beats abstracts by 0.932 to 0.247 on methods questions.
- **[doc/openalex.md](doc/openalex.md)** — the news-to-paper service: its corpus, its
  filters, and why there is no Semantic Scholar merge.
- **[doc/custommcp.md](doc/custommcp.md)** - named PDS collections, ingestion,
  isolated endpoints, and deploying the fourth service.
- **[doc/retrieval.md](doc/retrieval.md)** — how ranking works and what was measured to
  arrive at it, including the approaches that were tried and rejected.
- **[doc/evaluation.md](doc/evaluation.md)** — running a blind A/B of an LLM with and
  without ScienceMCP.

## Running it

Once per machine:

```bash
bash tools/gcr-prep.sh
```

Then per service, idempotent and safe to re-run — each pulls its own data, exports the
shared reranker if it is missing, and builds its own index only when stale:

```bash
source ~/mcp/env.sh
bash tools/sciencemcp-a100.sh prepare && bash tools/sciencemcp-a100.sh serve
bash tools/openalex-a100.sh   prepare && bash tools/openalex-a100.sh   serve
bash tools/mitmcp-a100.sh     prepare && bash tools/mitmcp-a100.sh     serve
bash tools/custommcp-a100.sh  prepare && bash tools/custommcp-a100.sh  serve
```

Run each service in its own terminal, or use the systemd units installed by
[tools/gcr-prep.sh](tools/gcr-prep.sh). `mcp-prepare` runs all four `prepare`s in
sequence and the four server units wait on it. The installer rewrites account
paths and prompts for credentials; do not commit credentials to the unit templates.

The server units are `mcp-science-server`, `mcp-openalex-server`, `mcp-mit-server`,
and `mcp-custom-server`. Their tokens are `SCIENCEPCM_TOKEN`, `OPENALEX_TOKEN`,
`MITMCP_TOKEN`, and `CUSTOMMCP_TOKEN`, respectively. Cloud preparation uses
`legopds_clienthash`, not a serving token.

## Layout

```
~/mcp/                     everything the box needs
   env.sh                  CUDA paths for ONNX Runtime
   venvs/                  sync, eval, lab, cuda12
    models/bge-reranker     model files shared by all four services
   data/                   pulled from the blob store, disposable
/datadisk/index/           built here, and only here
```

`/datadisk` is local NVMe and is wiped when the VM deallocates. The OpenAlex index is
larger than the free space on the OS disk, so there is no durable copy and a
deallocation costs a rebuild — `prepare` notices on its own, because the index stamp
goes with the disk.

## Source

```
src/SciencePcm.Core       JATS parsing and chunking
src/SciencePcm.Ingest     JATS -> Parquet
src/SciencePcm.Embed      ONNX inference, tokenizers, cross-encoders
src/SciencePcm.Index      the Lucene index and query, shared by all servers
src/SciencePcm.Server     ScienceMCP retrieval service and MCP tools
src/OpenAlex.*            OpenAlex ingest, index and server
src/CustomMcp.Ingest      named PDS download -> Parquet -> prepared indexes
src/CustomMcp.Server      /mcp/<name>, isolated indexes, one shared reranker
src/MitPcm.Ingester       dataset-specific MIT PDS repair and inspection
src/SciencePcm.Inspector  the browser console
eval/                     retrieval measurement, LLM judge, known-item tests
```

[LexicalIndex.cs](src/SciencePcm.Index/LexicalIndex.cs) is the piece worth reading
first: the services share its schema, its query construction and its scoring.
