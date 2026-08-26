"""Rerank an existing run file with the MedCPT cross-encoder.

A bi-encoder scores query and document independently, so it can only ever compare
compressed summaries of each. A cross-encoder reads the pair together and is far more
accurate, but it cannot search - it can only reorder candidates something else found.
That makes reranking a pure precision fix bounded by the first stage's recall: this
run's ceiling is its own recall@100.

Measuring it here in PyTorch first is deliberate. The C# reranker needs an ONNX export
of a model with a classification head, which is more export work than the encoders
were, and that is only worth doing once the gain is known.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="Run file to rerank")
    parser.add_argument("--qrels", required=True, type=Path, help="Supplies query_text")
    parser.add_argument("--articles", required=True, help="Glob for the abstracts Parquet")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument("--top", type=int, default=100, help="Candidates per query to rerank")
    parser.add_argument("--limit", type=int, default=None, help="Only the first n queries")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Some rerankers ship custom model code. Only for models you trust.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    runs = load_jsonl(args.run)
    if args.limit:
        runs = runs[: args.limit]
    query_text = {str(r["query_id"]): r["query_text"] for r in load_jsonl(args.qrels)}

    wanted = sorted({key for record in runs for key in record["hits"][: args.top]})
    print(f"queries    : {len(runs):,}")
    print(f"candidates : {len(wanted):,} unique articles")

    rows = duckdb.connect().execute(
        f"""
        SELECT openalex_id, COALESCE(title, ''), COALESCE(abstract, '')
        FROM read_parquet('{args.articles}')
        WHERE openalex_id IN (SELECT UNNEST(?))
        """,
        [wanted],
    ).fetchall()
    text_by_key = {row[0]: (row[1] + ". " + row[2] if row[1] else row[2]) for row in rows}
    print(f"text found : {len(text_by_key):,}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        .to(device)
        .eval()
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored_pairs = 0
    started = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for index, record in enumerate(runs, start=1):
            query_id = str(record["query_id"])
            candidates = [key for key in record["hits"][: args.top] if key in text_by_key]

            if not candidates:
                stream.write(json.dumps({"query_id": query_id, "hits": record["hits"]}) + "\n")
                continue

            query = query_text[query_id]
            scores = []

            for start in range(0, len(candidates), args.batch):
                chunk = candidates[start : start + args.batch]
                encoded = tokenizer(
                    [[query, text_by_key[key]] for key in chunk],
                    truncation=True,
                    padding=True,
                    max_length=args.max_tokens,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(dim=1)
                scores.extend(logits.float().cpu().tolist())
                scored_pairs += len(chunk)

            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            # Candidates without text keep their original relative order behind the reranked ones.
            tail = [key for key in record["hits"][: args.top] if key not in text_by_key]

            stream.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "hits": [candidates[i] for i in order] + tail,
                        "scores": [scores[i] for i in order],
                    }
                )
                + "\n"
            )

            if index % 50 == 0:
                print(f"  {index:,}/{len(runs):,}  ({scored_pairs:,} pairs)", end="\r")

    elapsed = time.perf_counter() - started
    print(f"\nScored {scored_pairs:,} pairs in {elapsed:,.0f}s ({scored_pairs / max(elapsed, 1e-9):,.0f} pairs/s)")
    print(f"Wrote {args.out}")
    print("Score it with: python eval/run_eval.py --retriever runfile --run <file> --qrels <qrels>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
