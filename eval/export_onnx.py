"""Export MedCPT encoders to ONNX for the C# inference path.

MedCPT is a *dual* encoder: articles and queries use different weights. Both are
exported. Confusing them produces plausible-looking but badly degraded retrieval.

Uses torch.onnx.export directly rather than optimum, which churns its API between
majors, and so the ONNX graph gets exactly the names the C# embedder binds to:
input_ids / attention_mask / token_type_ids -> last_hidden_state.

Two checks run automatically:
  * numerical parity between PyTorch and ONNX Runtime on the probe texts
  * tokenizer-parity.json, replayed later by SciencePcm.Embed --verify-tokenizer

Run with the lab environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"

# Deliberately awkward: hyphenation, Greek letters, gene symbols, superscripts, casing.
PROBE_TEXTS = [
    "Optogenetic silencing of CA1 pyramidal neurons impaired contextual fear recall.",
    "TNF-\u03b1 and IL-1\u03b2 levels were elevated in the prefrontal cortex.",
    "We used AAV9-hSyn-ChR2(H134R)-EYFP at 5 mW/mm2.",
    "The GRIK2 candidate gene revealed no variants of interest (p = 4.4 \u00d7 10\u22127).",
    "Patients with Parkinson's disease showed reduced [11C]PBR28 binding.",
    "\u03b2-amyloid plaques co-localised with GFAP+ astrocytes in 5xFAD mice.",
]

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]
OUTPUT_NAME = "last_hidden_state"


def build_wrapper(model):
    import torch

    class Encoder(torch.nn.Module):
        """Returns a bare tensor; the ONNX exporter cannot handle dataclass outputs."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            output = self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return output.last_hidden_state

    return Encoder(model).eval()


def export(model_id: str, destination: Path, max_tokens: int, opset: int) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoTokenizer

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} -> {destination}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    wrapper = build_wrapper(model)

    encoded = tokenizer(
        PROBE_TEXTS[:2],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    args = (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])

    onnx_path = destination / "model.onnx"
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}
    dynamic_axes[OUTPUT_NAME] = {0: "batch", 1: "sequence"}

    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            **kwargs,
        )

    tokenizer.save_pretrained(destination)
    write_vocab(tokenizer, destination)

    # Deliberately a different batch size and sequence length from the export sample:
    # TorchScript tracing can bake shapes in as constants, and reusing the export
    # inputs would hide exactly that failure.
    verify = tokenizer(
        PROBE_TEXTS,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    verify_args = (verify["input_ids"], verify["attention_mask"], verify["token_type_ids"])
    print(f"  export shape {tuple(args[0].shape)} -> verify shape {tuple(verify_args[0].shape)}")

    with torch.no_grad():
        reference = wrapper(*verify_args).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        [OUTPUT_NAME],
        {name: verify_args[i].numpy() for i, name in enumerate(INPUT_NAMES)},
    )[0]

    if reference.shape != actual.shape:
        raise RuntimeError(
            f"ONNX output shape {actual.shape} != PyTorch {reference.shape}; "
            "dynamic axes did not survive the export."
        )

    drift = float(np.abs(reference - actual).max())
    print(f"  torch vs onnxruntime max abs diff: {drift:.2e}")
    if drift > 1e-3:
        raise RuntimeError(f"ONNX export diverges from PyTorch (max abs diff {drift:.2e}).")

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  wrote: model.onnx ({size_mb:,.0f} MB), vocab.txt")


def write_vocab(tokenizer, destination: Path) -> None:
    """Fast tokenizers persist tokenizer.json, but the C# side needs a plain vocab.txt."""
    path = destination / "vocab.txt"
    if path.exists():
        return

    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    if ordered[0][1] != 0 or ordered[-1][1] != len(ordered) - 1:
        raise RuntimeError("Tokenizer vocabulary ids are not a contiguous range starting at 0.")

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for token, _ in ordered:
            stream.write(token + "\n")


def write_parity(model_id: str, destination: Path, max_tokens: int) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    samples = [
        {
            "text": text,
            "ids": tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens),
        }
        for text in PROBE_TEXTS
    ]

    payload = {
        "model_id": model_id,
        "max_tokens": max_tokens,
        "lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
        "samples": samples,
    }

    path = destination / "tokenizer-parity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote: tokenizer-parity.json ({len(samples)} probes, lower_case={payload['lower_case']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Root directory for exported models.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--article-only", action="store_true", help="Skip the query encoder.")
    args = parser.parse_args()

    targets = [(ARTICLE_MODEL, args.out / "medcpt-article")]
    if not args.article_only:
        targets.append((QUERY_MODEL, args.out / "medcpt-query"))

    for model_id, destination in targets:
        export(model_id, destination, args.max_tokens, args.opset)
        write_parity(model_id, destination, args.max_tokens)

    print()
    print("Point SciencePcm.Embed --model at medcpt-article for the corpus,")
    print("and at medcpt-query when embedding search queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
