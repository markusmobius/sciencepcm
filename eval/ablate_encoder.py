"""Ablate MedCPT article-encoding choices on a small pool, using PyTorch as ground truth.

Our C# pipeline deviates from MedCPT's reference usage in two independent ways:

1. It joins title and abstract as ``title + ". " + abstract`` (one segment, all
   token_type_ids zero). The reference passes them as a PAIR, giving
   ``[CLS] title [SEP] abstract [SEP]`` with segment ids 1 on the abstract.
2. It L2-normalises, so search ranks by cosine. The reference ranks by a raw dot
   product on unnormalised CLS vectors, which is not the same ordering.

Re-embedding 5.3M abstracts to test either guess costs hours, so this scores every
combination over a few hundred queries against a small candidate pool. Absolute
numbers are inflated by the small pool; only the comparison between variants matters.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb
import torch
from transformers import AutoModel, AutoTokenizer

from metrics import MetricAccumulator


def load_qrels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fetch_articles(articles_glob: str, keys: list[str], distractors: int) -> dict[str, tuple[str, str]]:
    """Title and abstract for the gold keys, plus a random pool of distractors."""
    connection = duckdb.connect()
    gold = connection.execute(
        f"""
        SELECT openalex_id, COALESCE(title, ''), COALESCE(abstract, '')
        FROM read_parquet('{articles_glob}')
        WHERE openalex_id IN (SELECT UNNEST(?))
        """,
        [keys],
    ).fetchall()

    noise = connection.execute(
        f"""
        SELECT openalex_id, COALESCE(title, ''), COALESCE(abstract, '')
        FROM read_parquet('{articles_glob}')
        WHERE abstract IS NOT NULL AND abstract <> ''
        USING SAMPLE {distractors} ROWS
        """
    ).fetchall()

    pool = {row[0]: (row[1], row[2]) for row in noise}
    pool.update({row[0]: (row[1], row[2]) for row in gold})
    return pool


@torch.inference_mode()
def encode(model, tokenizer, texts, max_length, batch_size, device, pair=False):
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        if pair:
            encoded = tokenizer(
                [t[0] for t in chunk],
                [t[1] for t in chunk],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        vectors.append(model(**encoded).last_hidden_state[:, 0, :].float().cpu())
        print(f"    {min(start + batch_size, len(texts)):,}/{len(texts):,}", end="\r")
    print(" " * 40, end="\r")
    return torch.cat(vectors)


def score(query_vectors, article_vectors, article_keys, qrels, normalise, k=100):
    q = torch.nn.functional.normalize(query_vectors, dim=1) if normalise else query_vectors
    a = torch.nn.functional.normalize(article_vectors, dim=1) if normalise else article_vectors

    accumulator = MetricAccumulator()
    similarities = q @ a.T
    top = similarities.topk(min(k, len(article_keys)), dim=1).indices

    for row, record in enumerate(qrels):
        ranked = [article_keys[i] for i in top[row].tolist()]
        accumulator.add(ranked, set(record["gold_article_keys"]))

    return accumulator.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--articles", required=True, help="Glob for the abstracts Parquet")
    parser.add_argument("--article-model", required=True, help="HF id or path, e.g. ncbi/MedCPT-Article-Encoder")
    parser.add_argument("--query-model", required=True, help="HF id or path, e.g. ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--distractors", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--query-max-tokens", type=int, default=64)
    parser.add_argument("--article-max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    qrels = load_qrels(args.qrels)
    random.shuffle(qrels)
    qrels = qrels[: args.queries]

    gold_keys = sorted({key for record in qrels for key in record["gold_article_keys"]})
    print(f"queries        : {len(qrels):,}")
    print(f"gold articles  : {len(gold_keys):,}")

    pool = fetch_articles(args.articles, gold_keys, args.distractors)
    missing = [key for key in gold_keys if key not in pool]
    if missing:
        print(f"WARNING: {len(missing):,} gold articles were not found in the Parquet.")

    article_keys = list(pool)
    print(f"candidate pool : {len(article_keys):,}")
    print()

    device = torch.device(args.device)
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    query_model = AutoModel.from_pretrained(args.query_model).to(device).eval()

    print("Encoding queries ...")
    query_vectors = encode(
        query_model,
        query_tokenizer,
        [record["query_text"] for record in qrels],
        args.query_max_tokens,
        args.batch,
        device,
    )
    del query_model
    torch.cuda.empty_cache()

    article_tokenizer = AutoTokenizer.from_pretrained(args.article_model)
    article_model = AutoModel.from_pretrained(args.article_model).to(device).eval()

    concat_texts = [
        (pool[key][0] + ". " + pool[key][1]) if pool[key][0] else pool[key][1] for key in article_keys
    ]
    pair_texts = [(pool[key][0], pool[key][1]) for key in article_keys]

    print("Encoding articles, concatenated (what we shipped) ...")
    concat_vectors = encode(
        article_model, article_tokenizer, concat_texts, args.article_max_tokens, args.batch, device
    )

    print("Encoding articles, title/abstract pair (reference usage) ...")
    pair_vectors = encode(
        article_model, article_tokenizer, pair_texts, args.article_max_tokens, args.batch, device, pair=True
    )

    variants = {
        "concat + cosine  (shipped)": (concat_vectors, True),
        "concat + dot": (concat_vectors, False),
        "pair + cosine": (pair_vectors, True),
        "pair + dot       (reference)": (pair_vectors, False),
    }

    results = {}
    print()
    print(f"{'variant':<30} {'ndcg@10':>9} {'recall@10':>10} {'mrr':>8}")
    print("-" * 60)
    for name, (vectors, normalise) in variants.items():
        summary = score(query_vectors, vectors, article_keys, qrels, normalise)
        results[name] = summary
        print(f"{name:<30} {summary['ndcg@10']:>9.4f} {summary['recall@10']:>10.4f} {summary['mrr']:>8.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
