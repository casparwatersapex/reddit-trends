from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.compute_embeddings import append_embeddings_stream


def test_append_embeddings_stream_writes_parquet(tmp_path: Path) -> None:
    output_path = tmp_path / "embeddings.parquet"
    rows = [
        {"post_id": "a1", "text": "One", "embedding": [0.1, 0.2]},
        {"post_id": "a2", "text": "Two", "embedding": [0.3, 0.4]},
    ]
    table = pa.Table.from_pylist(rows)
    schema = table.schema
    writer = pq.ParquetWriter(output_path, schema)
    try:
        append_embeddings_stream(writer, rows)
    finally:
        writer.close()

    df = pd.read_parquet(output_path)
    assert list(df["post_id"]) == ["a1", "a2"]
