from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


@dataclass(frozen=True)
class ValidationMetrics:
    total_rows: int
    missing_columns: list[str]
    null_counts: dict[str, int]
    invalid_date_count: int


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> ValidationResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ValidationResult(ok=False, errors=[f"Missing required columns: {missing}"])
    return ValidationResult(ok=True, errors=[])


def collect_validation_metrics(
    df: pd.DataFrame, required: Iterable[str], date_col: str = "date"
) -> ValidationMetrics:
    required_list = list(required)
    missing = [c for c in required_list if c not in df.columns]
    null_counts = {c: int(df[c].isna().sum()) for c in required_list if c in df.columns}
    invalid_date_count = int(df[date_col].isna().sum()) if date_col in df.columns else 0
    return ValidationMetrics(
        total_rows=len(df),
        missing_columns=missing,
        null_counts=null_counts,
        invalid_date_count=invalid_date_count,
    )


def require_values(
    df: pd.DataFrame,
    required: Iterable[str],
    date_col: str = "date",
    max_null_count: int = 0,
    max_invalid_date_count: int = 0,
) -> ValidationResult:
    metrics = collect_validation_metrics(df, required, date_col=date_col)
    errors: list[str] = []
    if metrics.missing_columns:
        errors.append(f"Missing required columns: {metrics.missing_columns}")

    null_exceeds = {
        col: count for col, count in metrics.null_counts.items() if count > max_null_count
    }
    if null_exceeds:
        errors.append(f"Null counts exceed limit (max {max_null_count}): {null_exceeds}")

    if metrics.invalid_date_count > max_invalid_date_count:
        errors.append(
            "Invalid date count exceeds limit "
            f"(max {max_invalid_date_count}): {metrics.invalid_date_count}"
        )

    return ValidationResult(ok=not errors, errors=errors)
