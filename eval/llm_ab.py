"""Run a paired LlmClient evaluation with and without the sciencemcp tool.

This script generates answers and durable run records. It deliberately does not grade
them; follow doc/evaluation.md to blind and score the resulting answers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from LlmClient.LlmLib import LlmFactory
from LlmClient.Models import Chat


SYSTEM_PROMPT = """You are answering a neuroscience research question. Use any tools
available to you when they would improve the answer. Base factual claims on
identifiable evidence. Distinguish findings from individual studies from general
consensus. Cite sources by DOI, PMID or PMCID when available. Do not invent citations.
If the available evidence is insufficient or protocols vary across studies, say so
explicitly. Give a concise answer followed by a Sources section."""

FORCE_TOOL_PROMPT = (
    "Before answering, use the sciencemcp tool to search for evidence relevant to "
    "this question."
)
CONDITIONS = ("no_tool", "sciencemcp")


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record.get("query_id", "")).strip()
            query_text = str(record.get("query_text", "")).strip()
            if not query_id or not query_text:
                raise ValueError(
                    f"{path}:{line_number} must contain nonempty query_id and query_text"
                )
            if query_id in seen:
                raise ValueError(f"duplicate query_id in {path}: {query_id}")
            seen.add(query_id)
            questions.append(record)

    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def load_completed(path: Path, model: str, forced: bool) -> set[tuple[str, int, str]]:
    completed: set[tuple[str, int, str]] = set()
    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if record.get("model") == model and bool(record.get("forced_retrieval")) == forced:
                completed.add(
                    (
                        str(record["query_id"]),
                        int(record.get("replicate", 1)),
                        str(record["condition"]),
                    )
                )
    return completed


async def ask(client, model: str, question: dict, condition: str, forced: bool) -> dict:
    tools = ["sciencemcp"] if condition == "sciencemcp" else None
    chat = Chat(responseSchema=None, model=model, tools=tools)
    chat.AddSystemMessage(SYSTEM_PROMPT)
    if forced and condition == "sciencemcp":
        chat.AddSystemMessage(FORCE_TOOL_PROMPT)
    chat.AddUserMessage(str(question["query_text"]))

    started = perf_counter()
    try:
        output = await client.Ask(chat, tags=["sciencepcm-eval", condition])
        answer = output.answer.ChatAnswer if output.answer else None
        error = str(output.error) if output.error is not None else None
    except Exception as exception:
        answer = None
        error = f"{type(exception).__name__}: {exception}"

    return {
        "answer": answer,
        "latency_ms": round((perf_counter() - started) * 1000),
        "error": error,
    }


async def run(args: argparse.Namespace) -> int:
    for variable in ("LLM_SERVER_URL", "LLM_USER_CODE", "LLM_CACHE"):
        if not os.getenv(variable):
            raise SystemExit(f"{variable} is not set")

    questions = load_questions(args.questions)
    rng = random.Random(args.seed)
    rng.shuffle(questions)

    if args.out.exists() and not args.resume:
        raise SystemExit(f"{args.out} already exists; use --resume or choose another --out")
    completed = load_completed(args.out, args.model, args.force_tool) if args.resume else set()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions) * args.repeats * len(CONDITIONS)
    done = len(completed)
    print(f"questions: {len(questions)}, repeats: {args.repeats}, answers: {total}")
    print(f"model: {args.model}")
    print(f"experiment: {'forced retrieval' if args.force_tool else 'tool availability'}")
    if completed:
        print(f"resuming with {len(completed)} existing records")

    factory = LlmFactory()
    client = await factory.create_client()
    try:
        with args.out.open("a", encoding="utf-8", buffering=1) as output_stream:
            for question in questions:
                query_id = str(question["query_id"])
                for replicate in range(1, args.repeats + 1):
                    condition_order = list(CONDITIONS)
                    rng.shuffle(condition_order)

                    for order, condition in enumerate(condition_order, start=1):
                        key = (query_id, replicate, condition)
                        if key in completed:
                            continue

                        result = await ask(
                            client, args.model, question, condition, args.force_tool
                        )
                        record = {
                            "run_id": args.run_id,
                            "query_id": query_id,
                            "query_text": question["query_text"],
                            "stratum": question.get("stratum"),
                            "replicate": replicate,
                            "condition_order": order,
                            "condition": condition,
                            "tool": "sciencemcp" if condition == "sciencemcp" else None,
                            "forced_retrieval": args.force_tool,
                            "model": args.model,
                            "system_prompt": SYSTEM_PROMPT,
                            **result,
                        }
                        output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_stream.flush()
                        completed.add(key)
                        done += 1
                        status = "error" if result["error"] else "ok"
                        print(
                            f"[{done}/{total}] {query_id} r{replicate} "
                            f"{condition}: {status} ({result['latency_ms']} ms)"
                        )
    finally:
        await client.Close()

    print(f"wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions-methods.jsonl"),
        help="JSONL with query_id and query_text fields",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Exact LlmClient model version")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--force-tool",
        action="store_true",
        help="Require sciencemcp in treatment (diagnostic experiment B)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing pairs to an existing compatible output file",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())