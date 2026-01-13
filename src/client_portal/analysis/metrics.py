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


def coerce_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        max_val = pd.to_numeric(series, errors="coerce").max()
        if pd.isna(max_val):
            return pd.to_datetime(series, errors="coerce")
        if max_val > 1_000_000_000_000:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def compute_topic_growth(
    df: pd.DataFrame,
    topic_col: str,
    date_col: str,
    window_days: int,
) -> pd.DataFrame:
    valid = df.dropna(subset=[date_col]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                topic_col,
                "recent_count",
                "prior_count",
                "growth_abs",
                "growth_pct",
            ]
        )
    max_date = valid[date_col].max()
    recent_start = max_date - pd.Timedelta(days=window_days)
    prior_start = recent_start - pd.Timedelta(days=window_days)

    recent = valid[valid[date_col] >= recent_start]
    prior = valid[(valid[date_col] >= prior_start) & (valid[date_col] < recent_start)]

    recent_counts = recent.groupby(topic_col).size().rename("recent_count")
    prior_counts = prior.groupby(topic_col).size().rename("prior_count")
    summary = pd.concat([recent_counts, prior_counts], axis=1).fillna(0).reset_index()
    summary["recent_count"] = summary["recent_count"].astype(int)
    summary["prior_count"] = summary["prior_count"].astype(int)
    summary["growth_abs"] = summary["recent_count"] - summary["prior_count"]
    summary["growth_pct"] = summary.apply(
        lambda row: (row["growth_abs"] / row["prior_count"]) if row["prior_count"] > 0 else None,
        axis=1,
    )
    return summary
