import pandas as pd

from scripts.keyword_growth import coerce_dates, compute_growth


def test_coerce_dates_numeric_seconds() -> None:
    series = pd.Series([1_700_000_000, 1_700_000_100])
    converted = coerce_dates(series)
    assert converted.dt.year.iloc[0] == 2023


def test_compute_growth_handles_empty() -> None:
    df = pd.DataFrame({"topic": [], "date": []})
    result = compute_growth(df, "topic", "date", window_days=30)
    assert result.empty


def test_compute_growth_counts() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-02-01", "2023-10-01", "2023-10-05"])
    df = pd.DataFrame({"topic": [1, 1, 1, 1, 1], "date": dates})
    result = compute_growth(df, "topic", "date", window_days=60)
    row = result.iloc[0]
    assert row["recent_count"] == 3
    assert row["prior_count"] == 1
