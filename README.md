# SciencePCM

Two MCP servers over scientific literature, both BM25 retrieval followed by a
cross-encoder rerank, served from one A100 box.

| | ScienceMCP | OpenAlex MCP |
| --- | --- | --- |
| corpus | 5.3M neuroscience abstracts + 15M full-text passages | 485M OpenAlex works |
| answers | what a study did and found | which paper a news article is talking about |
| endpoint | `www.sciencemcp.econlabs.org/mcp` | `www.openalexmcp.econlabs.org/mcp` |
| port | 8080 | 8081 |
| docs | [doc/sciencemcp.md](doc/sciencemcp.md) | [doc/openalex.md](doc/openalex.md) |

A browser console for poking at either lives at `www.mcptest.econlabs.org`.

## Documentation

- **[doc/provisioning.md](doc/provisioning.md)** — getting a GPU box ready: the right
  machine, .NET, Python environments, the CUDA 12 / cuDNN 9 dance ONNX Runtime needs.
- **[doc/operations.md](doc/operations.md)** — running it: tokens, the systemd units, the
  reverse tunnel and nginx, refreshing the data, what lives on which disk.
- **[doc/sciencemcp.md](doc/sciencemcp.md)** — the neuroscience service: two tiers, its
  tools, and why full text beats abstracts by 0.932 to 0.247 on methods questions.
- **[doc/openalex.md](doc/openalex.md)** — the news-to-paper service: its corpus, its
  filters, and why there is no Semantic Scholar merge.
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
```

Or install the units in `deploy/systemd/`, where `mcp-prepare` runs both `prepare`s in
sequence and the two server units wait on it.

## Layout

```
~/mcp/                     everything the box needs
   env.sh                  CUDA paths for ONNX Runtime
   venvs/                  sync, eval, lab, cuda12
   models/bge-reranker     shared by both services
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
src/SciencePcm.Index      the Lucene index and query, shared by both servers
src/SciencePcm.Server     ScienceMCP retrieval service and MCP tools
src/OpenAlex.*            OpenAlex ingest, index and server
src/SciencePcm.Inspector  the browser console
eval/                     retrieval measurement, LLM judge, known-item tests
```

`src/SciencePcm.Index/LexicalIndex.cs` is the piece worth reading first: both services
share its schema, its query construction and its scoring.
