"""Export MedCPT models to ONNX for the C# inference path.

MedCPT is a *dual* encoder: articles and queries use different weights. Both are
exported. Confusing them produces plausible-looking but badly degraded retrieval.

The cross-encoder is a third, different shape: it takes a query/passage PAIR and emits
a single relevance logit rather than an embedding, so it exports with a
sequence-classification head and a `logits` output whose only dynamic axis is batch.

Uses torch.onnx.export directly rather than optimum, which churns its API between
majors, and so the ONNX graph gets exactly the names the C# side binds to:
input_ids / attention_mask / token_type_ids -> last_hidden_state (or logits).

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
CROSS_MODEL = "ncbi/MedCPT-Cross-Encoder"

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
CROSS_OUTPUT_NAME = "logits"

# Query/passage pairs for the cross-encoder probes; one clear match per obvious mismatch.
PROBE_PAIRS = [
    ("What causes Friedreich's ataxia?", "Friedreich ataxia is caused by a GAA repeat expansion in FXN."),
    ("What causes Friedreich's ataxia?", "Wind turbine siting and its effect on local bird populations."),
    ("role of microglia in synaptic pruning", "Microglia engulf synapses via complement C1q and C3 during development."),
    ("role of microglia in synaptic pruning", "A survey of undergraduate attitudes to online lecture capture."),
    ("long COVID definition", "Post-acute sequelae of SARS-CoV-2 persisting beyond twelve weeks."),
    ("GDF15 expression increased in which states", "GDF15 rises with nutritional stress, pregnancy and metformin use."),
]


def build_wrapper(model, kind: str = "encoder"):
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
            return output.logits if kind == "cross" else output.last_hidden_state

    return Encoder(model).eval()


def encode_probes(tokenizer, kind: str, texts_or_pairs, max_length: int, pad_to_max: bool):
    padding = "max_length" if pad_to_max else True
    if kind == "cross":
        return tokenizer(
            [p[0] for p in texts_or_pairs],
            [p[1] for p in texts_or_pairs],
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    return tokenizer(
        texts_or_pairs,
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def export(model_id: str, destination: Path, max_tokens: int, opset: int, kind: str = "encoder") -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} -> {destination}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if kind == "cross":
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        output_name = CROSS_OUTPUT_NAME
        probes = PROBE_PAIRS
    else:
        model = AutoModel.from_pretrained(model_id).eval()
        output_name = OUTPUT_NAME
        probes = PROBE_TEXTS

    wrapper = build_wrapper(model, kind)

    encoded = encode_probes(tokenizer, kind, probes[:2], 64, pad_to_max=True)
    args = (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])

    onnx_path = destination / "model.onnx"
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}
    # The cross-encoder collapses the sequence into one score per pair, so batch is the
    # only axis that varies on its output.
    dynamic_axes[output_name] = {0: "batch"} if kind == "cross" else {0: "batch", 1: "sequence"}

    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=[output_name],
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
    verify = encode_probes(tokenizer, kind, probes, max_tokens, pad_to_max=False)
    verify_args = (verify["input_ids"], verify["attention_mask"], verify["token_type_ids"])
    print(f"  export shape {tuple(args[0].shape)} -> verify shape {tuple(verify_args[0].shape)}")

    with torch.no_grad():
        reference = wrapper(*verify_args).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        [output_name],
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

    if kind == "cross":
        # Odd probes are deliberate mismatches, so scores must alternate down.
        scores = actual.reshape(-1).tolist()
        for i in range(0, min(4, len(scores)), 2):
            print(f"  probe {i//2}: match {scores[i]:+.3f} vs mismatch {scores[i+1]:+.3f}")
            if scores[i] <= scores[i + 1]:
                raise RuntimeError("Cross-encoder scored a mismatched pair above its match.")

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


def write_parity(model_id: str, destination: Path, max_tokens: int, kind: str = "encoder") -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    samples = []

    if kind == "cross":
        # FastBertTokenizer has no pair overload, so the C# reranker assembles
        # [CLS] query [SEP] passage [SEP] by hand. Capture token_type_ids too: getting
        # the segment boundary wrong is silent and would quietly degrade every score.
        for query, passage in PROBE_PAIRS:
            encoded = tokenizer(query, passage, truncation=True, max_length=max_tokens)
            samples.append(
                {
                    "query": query,
                    "passage": passage,
                    "ids": encoded["input_ids"],
                    "token_type_ids": encoded["token_type_ids"],
                    "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
                }
            )
    else:
        for text in PROBE_TEXTS:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
            samples.append(
                {
                    "text": text,
                    "ids": ids,
                    "tokens": tokenizer.convert_ids_to_tokens(ids),
                }
            )

    payload = {
        "model_id": model_id,
        "kind": kind,
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
    parser.add_argument("--article-only", action="store_true", help="Skip the query and cross encoders.")
    parser.add_argument("--cross-only", action="store_true", help="Export only the cross-encoder.")
    parser.add_argument(
        "--cross-model",
        default=CROSS_MODEL,
        help="HuggingFace sequence-classification model used for cross-encoder export.",
    )
    parser.add_argument(
        "--cross-name",
        default="medcpt-cross",
        help="Output directory name for the cross-encoder.",
    )
    args = parser.parse_args()

    if args.cross_only:
        targets = [(args.cross_model, args.out / args.cross_name, "cross")]
    else:
        targets = [(ARTICLE_MODEL, args.out / "medcpt-article", "encoder")]
        if not args.article_only:
            targets.append((QUERY_MODEL, args.out / "medcpt-query", "encoder"))
            targets.append((args.cross_model, args.out / args.cross_name, "cross"))

    for model_id, destination, kind in targets:
        export(model_id, destination, args.max_tokens, args.opset, kind)
        write_parity(model_id, destination, args.max_tokens, kind)

    print()
    print("Point SciencePcm.Embed --model at medcpt-article for the corpus,")
    print("and at medcpt-query when embedding search queries.")
    print("medcpt-cross scores query/passage pairs and outputs 'logits', not embeddings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
