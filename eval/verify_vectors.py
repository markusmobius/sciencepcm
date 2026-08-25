"""Check that stored vectors really belong to the ids written beside them.

Shard alignment has only ever been checked structurally (byte counts against line
counts). That cannot detect an off-by-N shift, which is a live risk because the
passage run was interrupted and resumed. A shift would leave every count correct
while silently attaching each vector to the wrong document, and the symptom - weak
retrieval with healthy-looking scores - is exactly what we are chasing.

So: re-encode a sample of documents with PyTorch and compare against what is stored.
Matching vectors should agree to ~1.0. Anything near zero means the id lists and the
float rows are out of step.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb
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


def sample_stored(directory: Path, dimensions: int, count: int, seed: int):
    """Random (id, vector) pairs, spread across shards."""
    random.seed(seed)
    pairs = shard_pairs(directory)
    per_shard = max(1, count // len(pairs))
    sampled = []

    for vectors_path, ids_path in pairs:
        ids = [line.rstrip("\n") for line in ids_path.open(encoding="utf-8") if line.strip()]
        if not ids:
            continue

        rows = min(per_shard, len(ids))
        positions = sorted(random.sample(range(len(ids)), rows))

        with vectors_path.open("rb") as stream:
            for position in positions:
                stream.seek(position * dimensions * 4)
                buffer = np.frombuffer(stream.read(dimensions * 4), dtype=np.float32)
                sampled.append((ids[position], buffer.copy(), vectors_path.name, position))

        if len(sampled) >= count:
            break

    return sampled[:count]


@torch.inference_mode()
def encode(model_id, texts, max_length, batch_size, device, pair):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        if pair:
            encoded = tokenizer(
                [t[0] for t in chunk], [t[1] for t in chunk],
                truncation=True, padding=True, max_length=max_length, return_tensors="pt",
            )
        else:
            encoded = tokenizer(
                chunk, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        vectors.append(model(**encoded).last_hidden_state[:, 0, :].float().cpu())

    return torch.cat(vectors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--articles", required=True, help="Glob for the abstracts Parquet")
    parser.add_argument("--article-model", default="ncbi/MedCPT-Article-Encoder")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pair", action="store_true", help="Encode as a title/abstract pair instead of concatenated.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sampled = sample_stored(args.vectors, args.dimensions, args.sample, args.seed)
    print(f"sampled {len(sampled):,} stored vectors")

    keys = [item[0] for item in sampled]
    rows = duckdb.connect().execute(
        f"""
        SELECT openalex_id, COALESCE(title, ''), COALESCE(abstract, '')
        FROM read_parquet('{args.articles}')
        WHERE openalex_id IN (SELECT UNNEST(?))
        """,
        [keys],
    ).fetchall()
    text_by_key = {row[0]: (row[1], row[2]) for row in rows}

    present = [item for item in sampled if item[0] in text_by_key]
    print(f"found text for {len(present):,}")
    if not present:
        raise SystemExit("None of the sampled ids exist in the Parquet. Wrong --articles glob?")

    if args.pair:
        texts = [text_by_key[item[0]] for item in present]
    else:
        texts = [
            (text_by_key[item[0]][0] + ". " + text_by_key[item[0]][1])
            if text_by_key[item[0]][0]
            else text_by_key[item[0]][1]
            for item in present
        ]

    recomputed = encode(args.article_model, texts, args.max_tokens, args.batch, torch.device(args.device), args.pair)
    recomputed = torch.nn.functional.normalize(recomputed, dim=1)

    stored = torch.from_numpy(np.stack([item[1] for item in present]))
    stored_norms = stored.norm(dim=1)
    stored = torch.nn.functional.normalize(stored, dim=1)

    similarities = (stored * recomputed).sum(dim=1)
    ordered = similarities.sort().values

    print()
    print(f"stored norm mean : {stored_norms.mean():.4f}  (1.0 means they were L2-normalised)")
    print(f"cosine mean      : {similarities.mean():.4f}")
    print(f"cosine min       : {ordered[0]:.4f}")
    print(f"cosine p05       : {ordered[max(0, int(0.05 * len(ordered)))]:.4f}")
    print(f"cosine median    : {ordered[len(ordered) // 2]:.4f}")
    print(f"matching > 0.99  : {(similarities > 0.99).sum().item():,}/{len(similarities):,}")
    print()

    if similarities.mean() > 0.99:
        print("VERDICT: stored vectors match their ids. Alignment is sound.")
    elif similarities.mean() > 0.8:
        print("VERDICT: close but not exact. Suspect a text or truncation difference, not a shift.")
    else:
        print("VERDICT: stored vectors do NOT match their ids. The shards are misaligned.")

    worst = sorted(zip(similarities.tolist(), present), key=lambda pair: pair[0])[:5]
    print("\nworst five:")
    for similarity, item in worst:
        print(f"  {similarity:.4f}  {item[0]}  ({item[2]} row {item[3]:,})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "sampled": len(present),
                    "cosine_mean": float(similarities.mean()),
                    "cosine_min": float(ordered[0]),
                    "stored_norm_mean": float(stored_norms.mean()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
