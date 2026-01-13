from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from client_portal.pipeline.run import load_client_config, run_pipeline
from client_portal.pipeline.transform import apply_column_mapping, parse_types
from client_portal.pipeline.validate import collect_validation_metrics, require_values


@dataclass
class ExportMetrics:
    total_rows: int
    seen_columns: set[str]
    null_counts: dict[str, int]
    invalid_date_count: int


def init_metrics(required: list[str]) -> ExportMetrics:
    return ExportMetrics(
        total_rows=0,
        seen_columns=set(),
        null_counts={col: 0 for col in required},
        invalid_date_count=0,
    )


def update_metrics(metrics: ExportMetrics, chunk: pd.DataFrame, required: list[str]) -> None:
    metrics.total_rows += len(chunk)
    metrics.seen_columns.update(chunk.columns)
    for col in required:
        if col not in chunk.columns:
            chunk[col] = None
        metrics.null_counts[col] += int(chunk[col].isna().sum())
    if "date" in chunk.columns:
        metrics.invalid_date_count += int(chunk["date"].isna().sum())


def _stringify_value(value: object) -> object:
    if isinstance(value, (dict, list, set, tuple)):
        return json.dumps(value)
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def stringify_non_required_columns(chunk: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for col in chunk.columns:
        if col in required:
            continue
        chunk[col] = chunk[col].map(_stringify_value)
    return chunk


def _schema_with_nullable_strings(schema: pa.Schema) -> pa.Schema:
    fields = []
    for field in schema:
        if pa.types.is_null(field.type):
            fields.append(pa.field(field.name, pa.string(), nullable=True))
        else:
            fields.append(field.with_nullable(True))
    return pa.schema(fields)


def export_jsonl_chunked(
    input_path: Path,
    output_path: Path,
    required: list[str],
    mapping: dict[str, str],
    date_format: str | None,
    date_unit: str | None,
    chunksize: int,
    max_null_count: int,
    max_invalid_date_count: int,
) -> ExportMetrics:
    metrics = init_metrics(required)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    for chunk in pd.read_json(input_path, lines=True, chunksize=chunksize):
        chunk = apply_column_mapping(chunk, mapping)
        chunk = parse_types(
            chunk,
            date_col="date",
            date_format=date_format,
            date_unit=date_unit,
        )
        update_metrics(metrics, chunk, required)
        chunk = stringify_non_required_columns(chunk, required)

        if schema is None:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            schema = _schema_with_nullable_strings(table.schema)
            table = table.cast(schema, safe=False)
            writer = pq.ParquetWriter(output_path, schema)
        else:
            for col in schema.names:
                if col not in chunk.columns:
                    chunk[col] = None
            chunk = chunk[schema.names]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            table = table.cast(schema, safe=False)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    missing_columns = [c for c in required if c not in metrics.seen_columns]
    errors: list[str] = []
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
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
    if errors:
        raise ValueError("\n".join(errors))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical Reddit data.")
    parser.add_argument("input_path", help="Path to source JSONL/CSV/Parquet.")
    parser.add_argument("client_config_path", help="Path to client config.yml.")
    parser.add_argument(
        "--output-path",
        help="Destination Parquet file. Defaults to <input>_canonical.parquet.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="JSONL rows per chunk to keep memory stable.",
    )
    parser.add_argument(
        "--max-null-count",
        type=int,
        default=0,
        help="Maximum allowed nulls per required column.",
    )
    parser.add_argument(
        "--max-invalid-date-count",
        type=int,
        default=0,
        help="Maximum allowed invalid dates after parsing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else input_path.with_name(f"{input_path.stem}_canonical.parquet")
    )

    cfg = load_client_config(args.client_config_path)
    required = cfg.get("required_columns", [])
    mapping = cfg.get("column_mapping", {})
    date_format = cfg.get("date_format")
    date_unit = cfg.get("date_unit")

    if input_path.suffix.lower() in {".jsonl", ".json"}:
        metrics = export_jsonl_chunked(
            input_path=input_path,
            output_path=output_path,
            required=required,
            mapping=mapping,
            date_format=date_format,
            date_unit=date_unit,
            chunksize=args.chunksize,
            max_null_count=args.max_null_count,
            max_invalid_date_count=args.max_invalid_date_count,
        )
        missing_columns = [c for c in required if c not in metrics.seen_columns]
        print(
            {
                "rows": metrics.total_rows,
                "missing_columns": missing_columns,
                "null_counts": metrics.null_counts,
                "invalid_date_count": metrics.invalid_date_count,
            }
        )

        print(f"Wrote canonical export: {output_path}")
        return 0

    df = run_pipeline(input_path=input_path, client_config_path=args.client_config_path)
    metrics = collect_validation_metrics(df, required)
    print(
        {
            "rows": metrics.total_rows,
            "missing_columns": metrics.missing_columns,
            "null_counts": metrics.null_counts,
            "invalid_date_count": metrics.invalid_date_count,
        }
    )
    result = require_values(
        df,
        required,
        max_null_count=args.max_null_count,
        max_invalid_date_count=args.max_invalid_date_count,
    )
    if not result.ok:
        raise ValueError("\n".join(result.errors))
    df.to_parquet(output_path, index=False)
    print(f"Wrote canonical export: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
