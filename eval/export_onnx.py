"""Export MedCPT encoders to ONNX for the C# inference path.

MedCPT is a *dual* encoder: articles and queries use different weights. Both are
exported. Confusing them produces plausible-looking but badly degraded retrieval.

Also writes tokenizer-parity.json, containing token ids produced by the Python
tokenizer for a set of probe strings. SciencePcm.Embed --verify-tokenizer replays
these through the C# tokenizer. A mismatch there is the classic cause of "the
embeddings just aren't very good" and is otherwise invisible.

Run with the lab environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"

# Deliberately awkward: hyphenation, Greek letters, gene symbols, subscripts, casing.
PROBE_TEXTS = [
    "Optogenetic silencing of CA1 pyramidal neurons impaired contextual fear recall.",
    "TNF-\u03b1 and IL-1\u03b2 levels were elevated in the prefrontal cortex.",
    "We used AAV9-hSyn-ChR2(H134R)-EYFP at 5 mW/mm2.",
    "The GRIK2 candidate gene revealed no variants of interest (p = 4.4 \u00d7 10\u22127).",
    "Patients with Parkinson's disease showed reduced [11C]PBR28 binding.",
    "\u03b2-amyloid plaques co-localised with GFAP+ astrocytes in 5xFAD mice.",
]


def export(model_id: str, destination: Path) -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} -> {destination}")

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(destination)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(destination)

    if not (destination / "vocab.txt").exists():
        raise RuntimeError(
            f"vocab.txt was not written to {destination}. The C# tokenizer requires it."
        )

    onnx_files = sorted(p.name for p in destination.glob("*.onnx"))
    print(f"  wrote: {', '.join(onnx_files)} + vocab.txt")


def write_parity(model_id: str, destination: Path, max_tokens: int) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    samples = []
    for text in PROBE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
        samples.append({"text": text, "ids": ids})

    payload = {
        "model_id": model_id,
        "max_tokens": max_tokens,
        "lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
        "samples": samples,
    }

    path = destination / "tokenizer-parity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote: {path.name} ({len(samples)} probes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Root directory for exported models.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--article-only", action="store_true", help="Skip the query encoder.")
    args = parser.parse_args()

    targets = [(ARTICLE_MODEL, args.out / "medcpt-article")]
    if not args.article_only:
        targets.append((QUERY_MODEL, args.out / "medcpt-query"))

    for model_id, destination in targets:
        export(model_id, destination)
        write_parity(model_id, destination, args.max_tokens)

    print()
    print("Done. Point SciencePcm.Embed --model at the medcpt-article directory for the")
    print("corpus, and at medcpt-query when embedding search queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
