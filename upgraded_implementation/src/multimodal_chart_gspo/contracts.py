from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChartExample:
    question: str
    reference_answer: str
    image_path: str | None = None
    task: str = "unknown"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ChartExample":
        question, answer = str(data.get("question", "")).strip(), str(data.get("reference_answer", data.get("answer", ""))).strip()
        if not question or not answer:
            raise ValueError("question and reference_answer are required")
        return cls(question, answer, data.get("image_path"), str(data.get("task", "unknown")))
