from __future__ import annotations

import pandas as pd

from client_portal.pipeline.validate import collect_validation_metrics, require_values


def test_collect_validation_metrics_counts_nulls_and_dates() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["a1", "a2"],
            "date": [pd.Timestamp("2024-01-01"), pd.NaT],
        }
    )
    required = ["post_id", "date"]

    metrics = collect_validation_metrics(df, required)

    assert metrics.total_rows == 2
    assert metrics.missing_columns == []
    assert metrics.null_counts["date"] == 1
    assert metrics.invalid_date_count == 1


def test_require_values_flags_invalid_dates() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["a1", "a2"],
            "date": [pd.Timestamp("2024-01-01"), pd.NaT],
        }
    )
    required = ["post_id", "date"]

    result = require_values(df, required, max_null_count=0, max_invalid_date_count=0)

    assert result.ok is False
    assert any("Invalid date count exceeds limit" in err for err in result.errors)
