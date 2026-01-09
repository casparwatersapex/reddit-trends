from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from client_portal.pipeline.ingest import read_tabular
from client_portal.pipeline.transform import apply_column_mapping, parse_types
from client_portal.pipeline.validate import require_columns


def load_client_config(path: str | Path) -> dict:
    path = Path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_pipeline(input_path: str | Path, client_config_path: str | Path) -> pd.DataFrame:
    cfg = load_client_config(client_config_path)
    df = read_tabular(input_path)
    df = apply_column_mapping(df, cfg.get("column_mapping", {}))
    df = parse_types(df, date_col="date")

    required = cfg.get("required_columns", [])
    result = require_columns(df, required)
    if not result.ok:
        raise ValueError("\n".join(result.errors))

    return df
