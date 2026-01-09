from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> ValidationResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ValidationResult(ok=False, errors=[f"Missing required columns: {missing}"])
    return ValidationResult(ok=True, errors=[])
