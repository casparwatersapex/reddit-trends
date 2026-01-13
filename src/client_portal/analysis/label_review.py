from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    cleaned = cleaned.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    cleaned = cleaned.lstrip("'\"").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def extract_llm_value(text: str, key: str | None = None) -> str:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return ""
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
        if key is None:
            return cleaned
        value = payload.get(key, "")
        if value is None:
            return ""
        return str(value).strip()
    return cleaned
