# Provisioning the GPU box

Runbook for standing up the A100 machine from nothing. The reservation is 99 days
and renewable, so expect to do this again.

Everything runs from one script — `tools/gcr-prep.sh` — which is idempotent. Re-running
it is safe and cheap; each step checks before acting.

For an already-provisioned box, see [operations.md](operations.md), or
[adding CustomMcp](custommcp.md#deploy-on-the-existing-a100) for the new fourth service.

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

`--check` reports the machine and what it would install or build, but
changes nothing. **Always do this first** — it is how you catch a wrong host before
installing 3 GB of toolchain onto it.

## 4. Provision

```bash
bash tools/gcr-prep.sh
```

The Python dependencies and CUDA runtime dominate initial provisioning. Data
downloads, model export, and indexing happen in each service's `prepare`, afterwards.

## 5. Make CUDA visible in your shell

This is the step that is easy to forget later:

```bash
source ~/mcp/env.sh
```

**Every new shell needs this** before running anything with `--gpu`. Once you have
confirmed it works, make it permanent:

```bash
echo 'source ~/mcp/env.sh' >> ~/.bashrc
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
| Build | `dotnet build -c Release -p:UseGpu=true`. |
| Services | Offers to install and enable the shared prepare/tunnel units and four server units, without starting them. Prompts for four serving tokens and the cloud client hash. |

Flag: `--check`. Old data/model/index flags have moved out of this script; each
service owns its preparation. Read [operations.md](operations.md#secrets) before
installing units, especially when preserving existing client credentials.

Layout produced:

```
~/mcp/
  env.sh                                  source this in new shells
  venvs/{sync,eval,lab,cuda12}/
```

Service preparation then fills `~/mcp/data/`, exports the shared BGE reranker to
`~/mcp/models/bge-reranker`, and builds indexes under `/datadisk/index/`.

---

## Verification checklist

Provisioning worked if all of these hold:

- [ ] Hostname contains `GCRAZGDL`
- [ ] `gpu` line shows an A100
- [ ] The GPU build completes
- [ ] Service preparation completes, including tokenizer parity when available
- [ ] `source ~/mcp/env.sh` then a service's `serve` starts successfully with its GPU model

Note that the parity check is **tokenizer-only** and never loads the ONNX model, so it
proves nothing about CUDA. The build succeeding only proves the package resolved. The
service startup actually creates a CUDA session — if `LD_LIBRARY_PATH`
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

Service preparation needs HuggingFace only when the shared model is missing. If
access is blocked, export `BAAI/bge-reranker-v2-m3` with `tools/export_onnx.py` on a
connected machine, transfer the complete model directory to
`~/mcp/models/bge-reranker`, and rerun the service's `prepare`.

---


## Next

The machine is ready. Each service's `prepare` manages its data and indexes and
reuses the shared model files; running, exposing and refreshing are in
[operations.md](operations.md).

```bash
source ~/mcp/env.sh
bash tools/sciencemcp-a100.sh prepare
bash tools/openalex-a100.sh prepare
bash tools/mitmcp-a100.sh prepare
bash tools/custommcp-a100.sh prepare
```
