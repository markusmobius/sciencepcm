"""Rank metrics for article-level retrieval.

Judgements are binary: an article either is or is not in the BioASQ gold set.
Passage hits are collapsed to their parent article before scoring, because the
gold standard identifies documents rather than passages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def dedupe_preserving_order(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for index, key in enumerate(ranked, start=1):
        if key in gold:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, key in enumerate(ranked[:k], start=1) if key in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


@dataclass
class MetricAccumulator:
    cutoffs: tuple[int, ...] = (5, 10, 20, 50, 100)
    _recall: dict[int, list[float]] = field(default_factory=dict)
    _precision: dict[int, list[float]] = field(default_factory=dict)
    _ndcg: dict[int, list[float]] = field(default_factory=dict)
    _mrr: list[float] = field(default_factory=list)
    _latencies_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for k in self.cutoffs:
            self._recall.setdefault(k, [])
            self._precision.setdefault(k, [])
            self._ndcg.setdefault(k, [])

    def add(self, ranked: list[str], gold: set[str], latency_ms: float | None = None) -> None:
        ranked = dedupe_preserving_order(ranked)
        for k in self.cutoffs:
            self._recall[k].append(recall_at_k(ranked, gold, k))
            self._precision[k].append(precision_at_k(ranked, gold, k))
            self._ndcg[k].append(ndcg_at_k(ranked, gold, k))
        self._mrr.append(reciprocal_rank(ranked, gold))
        if latency_ms is not None:
            self._latencies_ms.append(latency_ms)

    @property
    def count(self) -> int:
        return len(self._mrr)

    def summary(self) -> dict[str, float | int]:
        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        result: dict[str, float | int] = {"queries": self.count, "mrr": mean(self._mrr)}
        for k in self.cutoffs:
            result[f"recall@{k}"] = mean(self._recall[k])
            result[f"precision@{k}"] = mean(self._precision[k])
            result[f"ndcg@{k}"] = mean(self._ndcg[k])

        if self._latencies_ms:
            ordered = sorted(self._latencies_ms)
            result["latency_ms_mean"] = mean(ordered)
            result["latency_ms_p50"] = ordered[len(ordered) // 2]
            result["latency_ms_p95"] = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

        return result


def format_summary(name: str, summary: dict[str, float | int]) -> str:
    lines = [f"=== {name} ===", f"queries: {summary['queries']}"]
    for key, value in summary.items():
        if key == "queries":
            continue
        lines.append(f"  {key:<20} {value:.4f}" if isinstance(value, float) else f"  {key:<20} {value}")
    return "\n".join(lines)
