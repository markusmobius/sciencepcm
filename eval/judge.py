"""Grade retrieval results with an LLM judge.

BioASQ cannot separate these systems. Its judgements cover ~10% of the corpus, so all
five configurations scored 0/10 on every sampled query while returning results that
were plainly on topic. Optimising against that metric would be optimising against an
artifact.

This pools the top-N from each system, has a judge grade every (query, document) pair
on a 0-3 scale, and computes graded nDCG. Pooling matters: each document is judged
once and reused across systems, so the comparison is fair and the cost scales with the
size of the union rather than the number of systems.

There is no local cache. The LLM server caches on the prompt AND the model, so a
re-run costs nothing and switching judge models cannot silently return the previous
model's grades.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
from pathlib import Path

import duckdb

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number", "enum": [0, 1, 2, 3]},
        "reason": {"type": "string"},
    },
    "required": ["relevance", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a neuroscience researcher assessing whether a {unit} is a useful answer to a "
    "colleague's question. Judge only the text shown. Grade on this scale:\n"
    "3 = directly answers the question, or reports the specific finding asked about\n"
    "2 = clearly relevant, covers the topic and would inform an answer\n"
    "1 = marginally related, shares subject matter but does not address the question\n"
    "0 = not relevant\n"
    "Judge relevance to the question asked, not the quality or recency of the work. "
    "A broad review can be a 3 if the question asks what something is."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked_grades: list[int], pooled_grades: list[int], k: int) -> float:
    ideal = dcg(sorted(pooled_grades, reverse=True)[:k])
    return dcg(ranked_grades[:k]) / ideal if ideal > 0 else 0.0


async def judge_pairs(pairs, texts, questions, model, workers, unit):
    """Grade every pair, returning {"query_id|key": grade}."""
    from LlmClient.LlmLib import LlmFactory
    from LlmClient.Models import Chat

    print(f"pairs to judge: {len(pairs):,} with {workers} workers on {model}")

    queue: asyncio.Queue = asyncio.Queue()
    for pair in pairs:
        queue.put_nowait(pair)

    factory = LlmFactory()
    grades: dict[str, int] = {}
    done = 0
    errors = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal done, errors
        client = await factory.create_client()
        try:
            while True:
                try:
                    query_id, key = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                chat = Chat(responseSchema=GRADE_SCHEMA, model=model)
                chat.AddSystemMessage(SYSTEM_PROMPT.format(unit=unit))
                chat.AddUserMessage(
                    f"Question: {questions[query_id]}\n\n"
                    f"{unit.capitalize()}:\n{texts[key]}\n\n"
                    "Return the relevance grade as JSON."
                )

                output = await client.Ask(chat, tags=["sciencepcm", "retrieval-judge"])

                async with lock:
                    done += 1
                    if output.error is not None:
                        errors += 1
                    else:
                        try:
                            parsed = json.loads(output.answer.ChatAnswer)
                            grades[f"{query_id}|{key}"] = int(parsed["relevance"])
                        except Exception:
                            errors += 1

                    if done % 200 == 0:
                        print(f"  judged {done:,}/{len(pairs):,}  ({errors} errors)")
        finally:
            await client.Close()

    await asyncio.gather(*[worker() for _ in range(workers)])
    print(f"  judged {done:,}/{len(pairs):,}  ({errors} errors)")
    return grades


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--queries", required=True, type=Path, help="JSONL supplying query text")
    parser.add_argument("--id-field", default="query_id")
    parser.add_argument("--text-field", default="query_text")
    parser.add_argument("--articles", required=True, help="Glob for the abstracts Parquet")
    parser.add_argument(
        "--unit",
        default="paper",
        choices=["paper", "passage"],
        help="What a hit is. Use passage when judging search_full_text.",
    )
    parser.add_argument("--top", type=int, default=10, help="Depth to judge and score.")
    parser.add_argument("--sample", type=int, default=150, help="Queries to judge.")
    parser.add_argument("--workers", type=int, default=32, help="The server runs 50 threads.")
    parser.add_argument("--model", default="gpt-5.6-sol_2026-07-09")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    for variable in ("LLM_SERVER_URL", "LLM_USER_CODE"):
        if not os.getenv(variable):
            raise SystemExit(f"{variable} is not set.")

    labels = args.label or [path.stem for path in args.run]
    if len(labels) != len(args.run):
        parser.error("--label must be given as often as --run")

    runs = [{str(r["query_id"]): r["hits"] for r in load_jsonl(path)} for path in args.run]

    # Runs from the live server carry the text they returned. Prefer it: a passage id
    # cannot be looked up in the abstracts Parquet, and even for abstracts it is the
    # text the system actually showed.
    texts: dict[str, str] = {}
    for path in args.run:
        for record in load_jsonl(path):
            for key, text in zip(record.get("hits", []), record.get("texts", [])):
                if text and key not in texts:
                    texts[key] = text
    questions = {
        str(r[args.id_field]): r[args.text_field]
        for r in load_jsonl(args.queries)
        if r.get(args.text_field)
    }

    shared = sorted(set(questions) & set.intersection(*[set(run) for run in runs]))
    random.seed(args.seed)
    chosen = random.sample(shared, min(args.sample, len(shared)))
    print(f"queries judged: {len(chosen):,} of {len(shared):,} shared")

    pooled: dict[str, list[str]] = {}
    for query_id in chosen:
        seen: list[str] = []
        for run in runs:
            for key in run[query_id][: args.top]:
                if key not in seen:
                    seen.append(key)
        pooled[query_id] = seen

    wanted = sorted({key for keys in pooled.values() for key in keys})
    missing = [key for key in wanted if key not in texts]

    if missing:
        rows = duckdb.connect().execute(
            f"""
            SELECT openalex_id, COALESCE(title, ''), COALESCE(abstract, '')
            FROM read_parquet('{args.articles}')
            WHERE openalex_id IN (SELECT UNNEST(?))
            """,
            [missing],
        ).fetchall()
        for row in rows:
            texts[row[0]] = f"{row[1]}\n\n{row[2][:2000]}"

    found = sum(1 for key in wanted if key in texts)
    print(f"documents pooled: {len(wanted):,}, text found for {found:,}")

    pairs = [(qid, key) for qid, keys in pooled.items() for key in keys if key in texts]

    grades = asyncio.run(judge_pairs(pairs, texts, questions, args.model, args.workers, args.unit))

    results = {}
    print()
    print(f"{'system':<28} {'nDCG@' + str(args.top):>9} {'mean grade':>11} {'% >=2':>7}")
    print("-" * 60)

    for label, run in zip(labels, runs):
        scores, means, strong = [], [], []
        for query_id in chosen:
            pool_grades = [grades.get(f"{query_id}|{k}") for k in pooled[query_id]]
            pool_grades = [g for g in pool_grades if g is not None]
            ranked = [grades.get(f"{query_id}|{k}", 0) for k in run[query_id][: args.top]]
            if not pool_grades:
                continue
            scores.append(ndcg_at_k(ranked, pool_grades, args.top))
            means.append(sum(ranked) / len(ranked) if ranked else 0)
            strong.append(sum(1 for g in ranked if g >= 2) / len(ranked) if ranked else 0)

        results[label] = {
            f"ndcg@{args.top}": sum(scores) / len(scores),
            "mean_grade": sum(means) / len(means),
            "fraction_relevant": sum(strong) / len(strong),
        }
        r = results[label]
        print(f"{label:<28} {r[f'ndcg@{args.top}']:>9.4f} {r['mean_grade']:>11.3f} {r['fraction_relevant']:>7.1%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"queries": len(chosen), "model": args.model, "systems": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
