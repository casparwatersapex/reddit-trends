from __future__ import annotations

import pandas as pd


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a tiny, stable summary table used across dashboards and reports."""
    return pd.DataFrame(
        {
            "rows": [len(df)],
            "min_date": [df["date"].min() if "date" in df.columns else None],
            "max_date": [df["date"].max() if "date" in df.columns else None],
            "total_value": [df["value"].sum() if "value" in df.columns else None],
        }
    )
