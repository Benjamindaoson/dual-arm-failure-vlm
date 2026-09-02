from __future__ import annotations

from dataclasses import dataclass
import re

from .answer_parser import extract_final_answer


@dataclass(frozen=True)
class RewardBreakdown:
    correctness: float
    format_reward: float
    total: float
    extracted_answer: str | None


def _normalize(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _recover_unformatted_answer(output: str) -> str:
    """Recover a simple terminal answer without granting a format reward."""
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[.!。！？]?\s*$", output.strip())
    return match.group(1) if match else output


def score_output(reference: str, output: str) -> RewardBreakdown:
    extracted = extract_final_answer(output)
    candidate = extracted if extracted is not None else _recover_unformatted_answer(output)
    correctness = 1.0 if _normalize(reference) == _normalize(candidate) else 0.0
    format_reward = 0.2 if extracted is not None else 0.0
    return RewardBreakdown(correctness, format_reward, correctness + format_reward, extracted)
