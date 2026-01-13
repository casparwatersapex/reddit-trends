from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.export_canonical import export_jsonl_chunked


def test_export_jsonl_chunked_writes_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {"id": "a1", "created_utc": 1_700_000_000, "title": "One"},
        {"id": "a2", "created_utc": 1_700_000_100, "title": "Two"},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output_path = tmp_path / "out.parquet"
    metrics = export_jsonl_chunked(
        input_path=input_path,
        output_path=output_path,
        required=["post_id", "date"],
        mapping={"id": "post_id", "created_utc": "date"},
        date_format=None,
        date_unit="s",
        chunksize=1,
        max_null_count=0,
        max_invalid_date_count=0,
    )

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert set(["post_id", "date"]).issubset(df.columns)
    assert metrics.total_rows == 2


def test_export_jsonl_chunked_normalizes_mixed_object_column(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {"id": "a1", "crosspost_parent_list": []},
        {"id": "a2", "crosspost_parent_list": False},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output_path = tmp_path / "out.parquet"
    export_jsonl_chunked(
        input_path=input_path,
        output_path=output_path,
        required=["post_id"],
        mapping={"id": "post_id"},
        date_format=None,
        date_unit=None,
        chunksize=1,
        max_null_count=0,
        max_invalid_date_count=0,
    )

    df = pd.read_parquet(output_path)
    values = df["crosspost_parent_list"].tolist()
    assert values[0] == "[]"
    assert values[1] in {"False", "false"}
