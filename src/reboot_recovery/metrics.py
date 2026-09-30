from __future__ import annotations

from collections import Counter, defaultdict
import json
import random
import re
from typing import Any, Callable, Iterable

from .rewards import parse_structured_prediction, score_prediction


def normalize_semantic_output(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```json\r?\n(.*?)\r?\n```", stripped, flags=re.DOTALL)
    return fenced.group(1) if fenced else stripped


def parse_state_prediction(text: str) -> dict[str, str] | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"state"} or not isinstance(value["state"], str):
        return None
    state = value["state"].strip().casefold()
    return {"state": state} if state in {"nominal", "failure", "recovery"} else None


def diagnose_predictions(
    rows: Iterable[dict[str, Any]], *, bootstrap_samples: int = 1000, bootstrap_seed: int = 42
) -> dict[str, Any]:
    items = list(rows)
    normalized = [{**row, "output": normalize_semantic_output(str(row.get("output", "")))} for row in items]
    states = Counter(
        (parse_structured_prediction(row["output"]) or {}).get("state", "__invalid__")
        for row in normalized
    )
    return {
        "protocol": evaluate_predictions(items, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed),
        "semantic": evaluate_predictions(normalized, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed),
        "outer_fence_removed_count": sum(
            str(original.get("output", "")).strip() != updated["output"]
            for original, updated in zip(items, normalized, strict=True)
        ),
        "semantic_predicted_state_counts": dict(states),
        "semantic_exact_count": sum(
            score_prediction(row["reference"], row["output"]).total == 1.0 for row in normalized
        ),
    }


def _macro_f1(gold: list[str], pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    # Macro-F1 is defined over reference classes. Invalid/unknown predictions
    # still create false negatives without becoming a synthetic support-0 class.
    labels = sorted(set(gold))
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


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_episode_ci(
    records: list[dict[str, Any]], *, samples: int, seed: int,
    parser: Callable[[str], dict[str, str] | None] = parse_structured_prediction,
) -> dict[str, Any] | None:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        episode = record.get("episode_index") or record.get("reference", {}).get("episode_index")
        if episode is None:
            return None
        by_episode[str(episode)].append(record)
    ids = sorted(by_episode)
    if not ids or samples < 1:
        return None
    rng = random.Random(seed)
    state_values: list[float] = []
    failure_values: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(ids) for _ in ids]
        state_gold: list[str] = []
        state_pred: list[str] = []
        for episode in chosen:
            for row in by_episode[episode]:
                ref = row["reference"]
                parsed = parser(str(row.get("output", "")))
                state_gold.append(str(ref["execution_state"]).strip().casefold())
                state_pred.append(parsed["state"] if parsed else "__invalid__")
        state_values.append(_macro_f1(state_gold, state_pred)[0])
        support = sum(value == "failure" for value in state_gold)
        correct = sum(gold == pred == "failure" for gold, pred in zip(state_gold, state_pred, strict=True))
        failure_values.append(correct / support if support else 0.0)
    return {
        "unit": "episode",
        "samples": samples,
        "seed": seed,
        "state_macro_f1_95ci": [_percentile(state_values, 0.025), _percentile(state_values, 0.975)],
        "failure_recall_95ci": [_percentile(failure_values, 0.025), _percentile(failure_values, 0.975)],
    }


def evaluate_predictions(
    rows: Iterable[dict[str, Any]], *, bootstrap_samples: int = 1000, bootstrap_seed: int = 42
) -> dict[str, Any]:
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

        anchor = str(ref.get("timing_slice", ref.get("anchor_kind", gold_state)))
        timing[anchor].append(gold_state == pred_state)

    phase_macro, phase_detail = _macro_f1(phase_gold, phase_pred)
    state_macro, state_detail = _macro_f1(state_gold, state_pred)
    mode_macro, mode_detail = _macro_f1(mode_gold, mode_pred)
    n = len(items)
    failure_support = sum(x == "failure" for x in state_gold)
    failure_correct = sum(g == "failure" and p == "failure" for g, p in zip(state_gold, state_pred, strict=True))
    recovery_support = sum(x == "recovery" for x in state_gold)
    recovery_correct = sum(g == "recovery" and p == "recovery" for g, p in zip(state_gold, state_pred, strict=True))
    nominal_support = sum(x == "nominal" for x in state_gold)
    pre_failure_false_positives = sum(
        gold == "nominal" and predicted == "failure"
        for gold, predicted in zip(state_gold, state_pred, strict=True)
    )

    return {
        "n": n,
        "json_valid_rate": valid / n if n else 0.0,
        "phase_macro_f1": phase_macro,
        "state_macro_f1": state_macro,
        "failure_recall": failure_correct / failure_support if failure_support else 0.0,
        "failure_correct": failure_correct,
        "failure_support": failure_support,
        "recovery_recall": recovery_correct / recovery_support if recovery_support else 0.0,
        "pre_failure_false_positive_rate": pre_failure_false_positives / nominal_support if nominal_support else 0.0,
        "failure_mode_macro_f1": mode_macro,
        "failure_mode_macro_f1_non_nominal": mode_macro,
        "phase_per_class": phase_detail,
        "state_per_class": state_detail,
        "failure_mode_per_class": mode_detail,
        "timing_slices": {
            name: sum(values) / len(values) for name, values in sorted(timing.items())
        },
        "gold_state_counts": dict(Counter(state_gold)),
        "episode_bootstrap": _bootstrap_episode_ci(items, samples=bootstrap_samples, seed=bootstrap_seed),
    }


def evaluate_state_predictions(
    rows: Iterable[dict[str, Any]], *, bootstrap_samples: int = 1000, bootstrap_seed: int = 42
) -> dict[str, Any]:
    items = list(rows)
    gold = [str(row["reference"]["execution_state"]).strip().casefold() for row in items]
    parsed = [parse_state_prediction(str(row.get("output", ""))) for row in items]
    pred = [value["state"] if value else "__invalid__" for value in parsed]
    macro, detail = _macro_f1(gold, pred)
    support = Counter(gold)
    correct = Counter(g for g, p in zip(gold, pred, strict=True) if g == p)
    return {
        "n": len(items),
        "json_valid_rate": sum(value is not None for value in parsed) / len(items) if items else 0.0,
        "state_macro_f1": macro,
        "failure_recall": correct["failure"] / support["failure"] if support["failure"] else 0.0,
        "failure_correct": correct["failure"],
        "failure_support": support["failure"],
        "nominal_recall": correct["nominal"] / support["nominal"] if support["nominal"] else 0.0,
        "recovery_recall": correct["recovery"] / support["recovery"] if support["recovery"] else 0.0,
        "pre_failure_false_positive_rate": sum(g == "nominal" and p == "failure" for g, p in zip(gold, pred, strict=True)) / support["nominal"] if support["nominal"] else 0.0,
        "state_per_class": detail,
        "gold_state_counts": dict(support),
        "predicted_state_counts": dict(Counter(pred)),
        "episode_bootstrap": _bootstrap_episode_ci(items, samples=bootstrap_samples, seed=bootstrap_seed, parser=parse_state_prediction),
    }
