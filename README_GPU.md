# Provisioning the GPU box

Runbook for standing up the A100 machine from nothing. The reservation is 99 days
and renewable, so expect to do this again.

Everything runs from one script — `tools/gcr-prep.sh` — which is idempotent. Re-running
it is safe and cheap; each step checks before acting.

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

Flags: `--check`, `--skip-pull`, `--skip-models`, `--skip-build`.

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
