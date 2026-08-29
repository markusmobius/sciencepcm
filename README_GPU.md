# Provisioning the GPU box

Runbook for standing up the A100 machine from nothing. The reservation is 99 days
and renewable, so expect to do this again.

Everything runs from one script — `tools/gcr-prep.sh` — which is idempotent. Re-running
it is safe and cheap; each step checks before acting.

To deploy the MCP server on an already-provisioned box, skip to
[Deploying the MCP server](#deploying-the-mcp-server).

---

## 1. Get on the right machine

**Check the hostname before anything else.**

| Prefix | Meaning |
| --- | --- |
| `GCRAZ`**`G`**`DL…` | GPU box — this is what you want |
| `GCRAZ`**`C`**`DL…` | CPU box — wrong, no GPU |

One letter apart, and it has cost us a wasted provisioning run. The reservation
portal lists a specific hostname but you may be assigned a different one in the same
class; what matters is the `G`.

A correct box looks like:

```
cores      : 24
memory     : 216 GB
gpu        : NVIDIA A100 80GB PCIe, 81920 MiB, 580.173.02
```

## 2. Set the cloud credential

The corpus is pulled from RemoteBlobStore, which needs a client hash:

```bash
export legopds_clienthash='<your hash>'
echo "export legopds_clienthash='<your hash>'" >> ~/.bashrc
```

Without it the script stops immediately.

## 3. Clone and dry-run

```bash
git clone https://github.com/markusmobius/sciencepcm ~/sciencepcm
cd ~/sciencepcm

bash tools/gcr-prep.sh --check
```

`--check` reports the machine, what it would install, and what it would pull, but
changes nothing. **Always do this first** — it is how you catch a wrong host before
installing 3 GB of toolchain onto it.

## 4. Provision

```bash
bash tools/gcr-prep.sh
```

Expect 20-40 minutes, dominated by the corpus download and the torch install.

## 5. Make CUDA visible in your shell

This is the step that is easy to forget later:

```bash
source ~/sciencepcm-data/env.sh
```

**Every new shell needs this** before running anything with `--gpu`. Once you have
confirmed it works, make it permanent:

```bash
echo 'source ~/sciencepcm-data/env.sh' >> ~/.bashrc
```

`env.sh` sets `PATH` for the local .NET install and `LD_LIBRARY_PATH` for the CUDA 12
libraries (see [gotchas](#gotchas)).

---

## What the script does

| Step | Action |
| --- | --- |
| Machine | Reports cores, RAM, disk, GPU. Fails if the client hash is unset. |
| Toolchain | Installs .NET 10 SDK into `~/.dotnet` and `uv` into `~/.local/bin`. |
| Environments | Creates `sync`, `eval`, `lab` venvs from `requirements/`. |
| CUDA runtime | Creates a `cuda12` venv with CUDA 12 + cuDNN 9 wheels, writes `env.sh`. |
| Corpus | Pulls abstracts, passages and questions (~8.4 GB) from the blob store. |
| Models | Exports MedCPT article + query encoders to ONNX from HuggingFace. |
| Build | `dotnet build -c Release -p:UseGpu=true`. |
| Tokenizer parity | Verifies C# tokenisation matches Python. **Must be 6/6.** |

Flags: `--check`, `--skip-pull`, `--force-pull`, `--skip-models`, `--skip-build`.

Layout produced:

```
~/sciencepcm-data/
  env.sh                                  source this in new shells
  sciencepcm/{abstracts,passages-2019-2025,questions}/
  models/{medcpt-article,medcpt-query}/
  venvs/{sync,eval,lab,cuda12}/
  .pulled-*                               markers so re-runs skip downloads
```

---

## Verification checklist

Provisioning worked if all of these hold:

- [ ] Hostname contains `GCRAZGDL`
- [ ] `gpu` line shows an A100
- [ ] Tokenizer parity reports **6/6 probes matched**
- [ ] `source ~/sciencepcm-data/env.sh` then a `--gpu` benchmark runs without a library error

Note that the parity check is **tokenizer-only** and never loads the ONNX model, so it
proves nothing about CUDA. The build succeeding only proves the package resolved. The
first thing that actually creates a CUDA session is the benchmark — if `LD_LIBRARY_PATH`
or cuDNN are wrong, that is where it surfaces.

---

## Gotchas

### The image has CUDA 13, ONNX Runtime wants CUDA 12

`nvidia-smi` shows a driver, and `/usr/local/cuda` holds CUDA 13. But ONNX Runtime
1.29's CUDA provider links against `libcudart.so.12`, `libcublas.so.12` and
`libcudnn.so.9`. cuDNN is not installed at all.

The script solves this by pip-installing the CUDA 12 wheels into a `cuda12` venv and
prepending their `lib` directories to `LD_LIBRARY_PATH`. Nothing system-wide changes,
no root is needed, and CUDA 13 is left alone. The driver is backward compatible, so
CUDA 12 userspace works fine.

Diagnose with:

```bash
ldconfig -p | grep -E 'libcudnn|libcublas|libcudart'
echo $LD_LIBRARY_PATH
```

### `MAXCORES` breaks the blob client

`RemoteBlobServer` parses its port from the first line of stdout, but prints a
`MAXCORES` warning there first when that variable is set. The script unsets it for its
own process. If you call `cloudstore.py` by hand on a machine with `MAXCORES` set,
`unset MAXCORES` first.

### uv venvs have no pip

Use `uv pip install --python <venv>/bin/python …`, not `python -m pip`.

### HuggingFace may be blocked

The script tests reachability before exporting models. If it fails, export on nerds21
instead, put the result in `mcpserver\__temp\models`, and run
`.\tools\nerds21-sync.ps1 -IncludeOptional` there — then re-run this script with
`--skip-models` after pulling.

---

## Refreshing when new papers arrive

Both idempotency checks compare **file names only**. A rebuilt corpus reuses the same
shard filenames with different contents, so a plain re-run would report success and
quietly keep the old data. Refreshing is therefore explicit:

```powershell
# nerds21 - re-ingests and re-uploads
.\tools\nerds21-sync.ps1 -Force
```

```bash
# here - ignores the .pulled-* markers
bash tools/gcr-prep.sh --force-pull --skip-models --skip-build
```

Then **re-embed from scratch**. There is deliberately no incremental embedding: at
~990 texts/s the whole corpus is about 1.5 h for abstracts and 1.8 h for passages, and
tracking which ids already have vectors, merging shards and keeping the index mapping
consistent would be far more machinery than three hours of GPU time is worth.

What needs re-embedding depends on what changed:

| Change | Re-embed |
| --- | --- |
| More full-text XML | passages only (~1.8 h) |
| Abstracts Parquet regenerated | abstracts only (~1.5 h) |
| Chunker settings changed | passages, and any evaluation tied to them |

Changing chunk size or the section rules shifts every chunk id, which invalidates the
whole passage tier rather than extending it.

---

## After provisioning

Benchmark first, to find the batch size that saturates the GPU:

```bash
source ~/sciencepcm-data/env.sh
cd ~/sciencepcm

dotnet run --project src/SciencePcm.Embed -c Release -p:UseGpu=true -- \
  --model ~/sciencepcm-data/models/medcpt-article \
  --input "~/sciencepcm-data/sciencepcm/abstracts/part-*.parquet" \
  --benchmark --benchmark-texts 20000 --gpu --workers 4 --batch 256
```

On GPU the tuning knobs invert compared with CPU: `--intra-threads` stops mattering,
batch size matters much more, and a handful of workers is enough to keep the device fed
while the CPU tokenises.

**`--batch 256` with `--workers 4` is the validated setting: 953 texts/s, 1.3% padding
waste.** Do not raise the batch much beyond that. Attention memory is quadratic in
sequence length, so at batch 1024 and 512 tokens a single attention tensor is
`1024 x 12 heads x 512 x 512 x 4 bytes` = 12 GB, and four concurrent workers exhaust
even an 80 GB card. It fails partway through, once length-sorted batches reach the
long tail.

For reference, CPU on nerds21 (338 cores) managed 35 texts/s.

Then embed the abstract tier:

```bash
dotnet run --project src/SciencePcm.Embed -c Release -p:UseGpu=true -- \
  --model ~/sciencepcm-data/models/medcpt-article \
  --input "~/sciencepcm-data/sciencepcm/abstracts/part-*.parquet" \
  --out ~/sciencepcm-data/vectors/abstracts --gpu --workers 4 --batch 256
```

---

## Deploying the MCP server

The serving path is **BM25 (Lucene.NET) then MedCPT cross-encoder reranking**. There is
no dense retrieval in it: an LLM judge over 150 questions scored BM25+rerank at 0.771
graded nDCG@10 against 0.765 for dense-fused+rerank and 0.676 for BM25 alone, so the
HNSW index and the embedding vectors are not on the query path at all. They stay
archived in the blob store in case the full-text tier behaves differently.

### 1. Build the served index

Distinct from the eval index: these store text plus bibliographic metadata because the
reranker needs the passage text at query time and MCP results must identify their source.
The abstract Parquet already contains DOI, PMCID, publication date, journal, citations
and open-access links. The passage build joins the existing JATS `articles-part` shards
to `chunks-part` by article key; it does not rerun XML ingest.

```bash
source ~/sciencepcm-data/env.sh
cd ~/sciencepcm

dotnet run --project src/SciencePcm.Lexical -c Release -- build \
  --input "$HOME/sciencepcm-data/sciencepcm/abstracts/*.parquet" \
  --schema abstracts \
  --out ~/sciencepcm-data/index/abstracts-bm25 \
  --threads 16 --ram-buffer 2048
```

Build the enriched full-text passage index:

```bash
dotnet run --project src/SciencePcm.Lexical -c Release -- build \
  --input "$HOME/sciencepcm-data/sciencepcm/passages-2019-2025/chunks-part-*.parquet" \
  --metadata "$HOME/sciencepcm-data/sciencepcm/passages-2019-2025/articles-part-*.parquet" \
  --schema chunks \
  --out ~/sciencepcm-data/index/passages-bm25 \
  --threads 16 --ram-buffer 2048
```

Alternatively, the provisioning helper rebuilds both indexes with the correct metadata:

```bash
bash tools/gcr-prep.sh --force-index --skip-pull --skip-models
```

### 2. Export the reranker

Only needed once, and already done by `gcr-prep.sh` if
`~/sciencepcm-data/models/bge-reranker` exists.

```bash
~/sciencepcm-data/venvs/lab/bin/python tools/export_onnx.py \
  --out ~/sciencepcm-data/models --reranker BAAI/bge-reranker-v2-m3
```

**BGE is the served reranker.** An LLM judge over 400 questions put it at 0.851 graded
nDCG@10 against MedCPT's 0.781; on the same pooled run `bge-reranker-base` scored 0.759,
so size and training data decided this, not recency. MedCPT remains selectable - export
it with `--cross-only` and point `--cross-encoder` at `medcpt-cross`. The server picks
the tokenizer from the directory: a `tokenizer.onnx` means SentencePiece, otherwise
WordPiece.

The export self-checks: PyTorch vs ONNX Runtime parity, a different batch shape at
verification than at export, and matched query/passage pairs must outscore deliberately
mismatched ones.

Pairs are assembled in C#, so verify that too before serving:

```bash
  dotnet run --project src/SciencePcm.Embed -c Release -- \
  --model ~/sciencepcm-data/models/bge-reranker \
  --verify-pairs ~/sciencepcm-data/models/bge-reranker/tokenizer-parity.json
```

### 3. Set the token

In `~/.bashrc`, so it survives reconnects:

```bash
export SCIENCEPCM_TOKEN="a-long-random-string"
```

Without it the server starts anyway and prints `auth : OPEN - no token set`. It listens
on the LAN, so treat that warning as real. The token is a shared secret over plain HTTP,
not transport security - anything beyond a trusted network needs TLS in front.

### 4. Run it

```bash
screen -S mcp
source ~/sciencepcm-data/env.sh
cd ~/sciencepcm

dotnet run --project src/SciencePcm.Server -c Release -p:UseGpu=true -- \
  --index ~/sciencepcm-data/index/abstracts-bm25 \
  --passage-index ~/sciencepcm-data/index/passages-bm25 \
  --cross-encoder ~/sciencepcm-data/models/bge-reranker \
  --gpu \
  --urls http://0.0.0.0:8080
```

`Ctrl+A` `D` to detach. Confirm with:

```bash
curl -s localhost:8080/health
```

The service loads the index and the model at startup rather than on first request, so a
wrong path fails immediately instead of on someone's first question.

### 5. Point an LLM at it

### 5. Expose it publicly

The GPU box takes no inbound connections, so a reverse SSH tunnel makes it appear on the
relay (`www.llmserver.econlabs.org`), where nginx terminates TLS.

```bash
screen -S tunnel
cd ~/sciencepcm
./tools/mcp-tunnel.sh
```

One SSH connection, one forward:

| relay port | local port | serves |
| --- | --- | --- |
| 9201 | 8080 | `https://www.sciencemcp.econlabs.org` — production |

By default the script forwards both SciencePCM (`9201:8080`) and OpenAlex MCP
(`9202:8081`). Override the `FORWARDS` variable to run only a subset. The nginx vhosts live in
`deploy/nginx/`; each file carries its own install instructions in a header comment.
`www.mcptest.econlabs.org` is a staging endpoint served by a process running on the
relay itself, so it needs no forward here.

`-R` binds to the relay's loopback, so 9201 is never directly exposed — only nginx can
reach it. Verify with `ss -tlnp | grep 9201` on the relay.

### 6. Point an LLM at it

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

Check the chain in order, so a failure names its own hop:

```bash
curl -s localhost:8080/health                          # on the GPU box
curl -s localhost:9201/health                          # on the relay: tunnel up?
curl -s https://www.sciencemcp.econlabs.org/health     # anywhere: nginx and TLS
```

`/health` needs no token, so it separates "unreachable" from "wrong token".

### 7. Try it by hand

`src/SciencePcm.Inspector` is a local console for exercising any MCP server — server
picker, per-server bearer tokens in localStorage, tool browser and a result view.

```bash
dotnet run --project src/SciencePcm.Inspector -c Release
```

Then open <http://localhost:5173>. It proxies requests through itself by default, which
sidesteps CORS on servers that send no such headers.

### Tools exposed

| tool | purpose |
|---|---|
| `search_literature` | Natural-language question, optional `yearMin`/`yearMax`, optional `fast` to skip reranking. |
| `get_paper` | Full abstract by article key. |
| `corpus_stats` | Corpus scope and its caveats. |

`search_literature` wants a full question, not keywords - the cross-encoder reads the
question and the abstract together, so phrasing carries signal. Results are deduplicated
by title, because the corpus holds the same paper under several OpenAlex ids.

### Sharing the GPU

The cross-encoder scores `--rerank-candidates` pairs per query (100 by default), which
dominates latency. On CPU that was 0.5-1.0 s per 50 pairs, which is why `--gpu` matters.

To share the card with the LegoPCM hourly job, cap the allocator:

```bash
--gpu-mem-limit-gb 8
```

To trade quality for latency, lower `--rerank-candidates` to 50, or let callers pass
`fast: true`.

---

## Before the reservation lapses

The box is a working copy; nerds21 and the blob store are the durable side. Push
anything expensive back before you lose it:

```bash
~/sciencepcm-data/venvs/sync/bin/python tools/cloudstore.py push \
  --local ~/sciencepcm-data/vectors/abstracts \
  --cloud sciencepcm/abstract-vectors \
  --report ~/sciencepcm-data/vectors/abstracts/embed-report.json
```

`push` keys the artifact by a hash of the pipeline configuration read from
`embed-report.json`, so different models or token limits become different artifacts
rather than silent overwrites.

Nothing else on the box is worth saving — it all rebuilds from this script.
