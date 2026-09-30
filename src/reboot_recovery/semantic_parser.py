from __future__ import annotations

import re


_OUTER_FENCE = re.compile(r"```(?:json)?\r?\n(.*?)\r?\n```", re.DOTALL)


def normalize_semantic_output(text: str) -> str:
    """Remove only outer whitespace and one complete lowercase/unlabelled fence."""
    stripped = text.strip()
    match = _OUTER_FENCE.fullmatch(stripped)
    return match.group(1) if match else stripped
