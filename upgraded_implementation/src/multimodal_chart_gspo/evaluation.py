from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .rewards import score_output


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, float | int | dict[str, float]]:
    rows = list(records)
    scored = [score_output(str(row["reference_answer"]), str(row["output"])) for row in rows]
    by_task: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scored, strict=True):
        by_task[str(row.get("task", "unknown"))].append(score.correctness)
    total = len(scored)
    return {
        "total": total,
        "accuracy": sum(score.correctness for score in scored) / total if total else 0.0,
        "format_rate": sum(score.format_reward > 0 for score in scored) / total if total else 0.0,
        "task_accuracy": {task: sum(values) / len(values) for task, values in by_task.items()},
    }
