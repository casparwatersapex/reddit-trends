from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def compute_growth(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute growth for L3 keyword topics.")
    parser.add_argument("canonical_path", help="Canonical parquet with posts and dates.")
    parser.add_argument("l3_clusters_path", help="Parquet with L3 topics + parent_topic.")
    parser.add_argument("keywords_path", help="CSV with keyword ideas for L3 topics.")
    parser.add_argument("--output-summary", default="data/keywords_openai_l3_growth.csv")
    parser.add_argument("--output-timeseries", default="data/keywords_openai_l3_timeseries.csv")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--window-days", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_df = pd.read_parquet(args.canonical_path)
    l3_df = pd.read_parquet(args.l3_clusters_path)
    keywords_df = pd.read_csv(args.keywords_path)

    merged = l3_df.merge(canonical_df, on="post_id", how="inner")
    merged[args.date_col] = coerce_dates(merged[args.date_col])
    merged = merged.loc[merged["topic"] != -1]

    timeseries = (
        merged.dropna(subset=[args.date_col])
        .assign(month=lambda df: df[args.date_col].dt.to_period("M").dt.to_timestamp())
        .groupby(["topic", "parent_topic", "month"])
        .size()
        .rename("post_count")
        .reset_index()
    )
    timeseries_path = Path(args.output_timeseries)
    timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(timeseries_path, index=False)

    growth = compute_growth(merged, "topic", args.date_col, args.window_days)
    summary = growth.merge(
        keywords_df,
        left_on="topic",
        right_on="l3_topic",
        how="left",
    )
    summary = summary.rename(
        columns={
            "topic": "l3_topic",
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote growth summary: {summary_path}")
    print(f"Wrote growth timeseries: {timeseries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
