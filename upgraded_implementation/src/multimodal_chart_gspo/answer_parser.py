from __future__ import annotations

import re


def extract_final_answer(text: str) -> str | None:
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if tagged:
        return " ".join(tagged[-1].split())
    labelled = re.search(r"(?:final\s*answer|答案)\s*[:：]\s*([^\n]+)", text, flags=re.I)
    return " ".join(labelled.group(1).split()) if labelled else None
