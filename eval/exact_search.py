"""Exact brute-force search over the stored vectors, to measure what HNSW is costing.

The encoder ablation scores an exact ranking over a small pool, while the deployed
system scores an approximate ranking over 5.3M vectors. Those differ in two ways at
once, so a disappointing end-to-end number cannot be attributed to either. This runs
the same queries exactly, on the very same stored vectors, so the only remaining
difference against the deployed run is the ANN index itself.

Vectors stream to the GPU in slabs; only the running top-k is kept, so peak memory is
set by --slab rather than by the size of the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def shard_pairs(directory: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for vectors in sorted(directory.glob("vectors-part-*.f32")):
        ids = directory / vectors.name.replace("vectors-part-", "ids-part-").replace(".f32", ".txt")
        if not ids.exists():
            raise FileNotFoundError(f"No id file for {vectors.name}")
        pairs.append((vectors, ids))
    if not pairs:
        raise FileNotFoundError(f"No vectors-part-*.f32 in {directory}")
    return pairs


def load_qrels(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return records[:limit] if limit else records


@torch.inference_mode()
def encode_queries(model_id, texts, max_length, batch_size, device, normalise):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        vectors.append(model(**encoded).last_hidden_state[:, 0, :].float())

    result = torch.cat(vectors)
    del model
    torch.cuda.empty_cache()
    return torch.nn.functional.normalize(result, dim=1) if normalise else result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, type=Path, help="Directory of vectors-part-*.f32")
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--query-model", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--run", required=True, type=Path, help="Run file to write")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Only the first n queries")
    parser.add_argument("--slab", type=int, default=1_000_000, help="Vectors per GPU slab")
    parser.add_argument("--query-max-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--raw-queries",
        action="store_true",
        help="Do not L2-normalise the query vectors. The stored article vectors are already "
        "normalised, so this only makes sense when checking scoring behaviour.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    qrels = load_qrels(args.qrels, args.limit)
    print(f"queries : {len(qrels):,}")

    queries = encode_queries(
        args.query_model,
        [record["query_text"] for record in qrels],
        args.query_max_tokens,
        args.batch,
        device,
        normalise=not args.raw_queries,
    )
    print(f"encoded : {tuple(queries.shape)}")

    best_scores = torch.full((len(qrels), args.k), -float("inf"), device=device)
    best_global = torch.zeros((len(qrels), args.k), dtype=torch.long, device=device)

    ids: list[str] = []
    offset = 0
    row_bytes = args.dimensions * 4

    for vectors_path, ids_path in shard_pairs(args.vectors):
        shard_ids = [line.rstrip("\n") for line in ids_path.open(encoding="utf-8") if line.strip()]
        ids.extend(shard_ids)

        with vectors_path.open("rb") as stream:
            position = 0
            while position < len(shard_ids):
                take = min(args.slab, len(shard_ids) - position)
                buffer = np.frombuffer(stream.read(take * row_bytes), dtype=np.float32)
                slab = torch.from_numpy(buffer.reshape(take, args.dimensions).copy()).to(device)

                scores = queries @ slab.T
                top = min(args.k, take)
                slab_scores, slab_indices = scores.topk(top, dim=1)

                merged_scores = torch.cat([best_scores, slab_scores], dim=1)
                merged_global = torch.cat([best_global, slab_indices + offset + position], dim=1)
                best_scores, order = merged_scores.topk(args.k, dim=1)
                best_global = merged_global.gather(1, order)

                position += take
                del slab, scores

        offset += len(shard_ids)
        print(f"  scanned {offset:,}", end="\r")

    print(f"  scanned {offset:,}")
    if offset != len(ids):
        raise SystemExit(f"Vector count {offset:,} does not match id count {len(ids):,}")

    args.run.parent.mkdir(parents=True, exist_ok=True)
    with args.run.open("w", encoding="utf-8") as stream:
        global_indices = best_global.cpu().tolist()
        scores = best_scores.cpu().tolist()
        for row, record in enumerate(qrels):
            hits, seen = [], set()
            for index in global_indices[row]:
                key = ids[index].split("#", 1)[0]
                if key not in seen:
                    seen.add(key)
                    hits.append(key)
            stream.write(
                json.dumps({"query_id": record["query_id"], "hits": hits, "scores": scores[row][: len(hits)]})
                + "\n"
            )

    print(f"\nWrote {args.run}")
    print("Score it with: python eval/run_eval.py --retriever runfile --run <file> --qrels <qrels>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
