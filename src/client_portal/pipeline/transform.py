from __future__ import annotations

import pandas as pd


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename input columns to canonical names."""
    return df.rename(columns=mapping).copy()


def parse_types(
    df: pd.DataFrame,
    date_col: str = "date",
    date_format: str | None = None,
    date_unit: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        if date_unit:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce", unit=date_unit)
        else:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce", format=date_format)
    return out
