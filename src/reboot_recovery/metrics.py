from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .rewards import parse_structured_prediction


def _macro_f1(gold: list[str], pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    labels = sorted(set(gold) | set(pred))
    detail: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        detail[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(g == label for g in gold)),
        }
        f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), detail


def evaluate_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    valid = 0
    phase_gold: list[str] = []
    phase_pred: list[str] = []
    state_gold: list[str] = []
    state_pred: list[str] = []
    mode_gold: list[str] = []
    mode_pred: list[str] = []
    timing: dict[str, list[bool]] = defaultdict(list)

    for row in items:
        ref = row["reference"]
        parsed = parse_structured_prediction(str(row.get("output", "")))
        gold_phase = str(ref["phase_name"]).strip().casefold()
        gold_state = str(ref["execution_state"]).strip().casefold()
        gold_mode = str(ref["failure_mode"]).strip().casefold()
        if parsed is None:
            pred_phase = pred_state = pred_mode = "__invalid__"
        else:
            valid += 1
            pred_phase = parsed["phase"]
            pred_state = parsed["state"]
            pred_mode = parsed["failure_mode"]

        phase_gold.append(gold_phase)
        phase_pred.append(pred_phase)
        state_gold.append(gold_state)
        state_pred.append(pred_state)
        if gold_state != "nominal":
            mode_gold.append(gold_mode)
            mode_pred.append(pred_mode)

        anchor = str(ref.get("anchor_kind", gold_state))
        timing[anchor].append(gold_state == pred_state)

    phase_macro, phase_detail = _macro_f1(phase_gold, phase_pred)
    state_macro, state_detail = _macro_f1(state_gold, state_pred)
    mode_macro, mode_detail = _macro_f1(mode_gold, mode_pred)
    n = len(items)
    failure_support = sum(x == "failure" for x in state_gold)
    failure_correct = sum(g == "failure" and p == "failure" for g, p in zip(state_gold, state_pred, strict=True))

    return {
        "n": n,
        "json_valid_rate": valid / n if n else 0.0,
        "phase_macro_f1": phase_macro,
        "state_macro_f1": state_macro,
        "failure_recall": failure_correct / failure_support if failure_support else 0.0,
        "failure_mode_macro_f1_non_nominal": mode_macro,
        "phase_per_class": phase_detail,
        "state_per_class": state_detail,
        "failure_mode_per_class": mode_detail,
        "timing_state_accuracy": {
            name: sum(values) / len(values) for name, values in sorted(timing.items())
        },
        "gold_state_counts": dict(Counter(state_gold)),
    }
