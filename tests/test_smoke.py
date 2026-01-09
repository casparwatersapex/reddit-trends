from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from client_portal.pipeline.run import run_pipeline


def test_pipeline_smoke(tmp_path: Path):
    # Create a minimal CSV matching the example client mapping
    p = tmp_path / "input.csv"
    p.write_text("Date,Amount\n2024-01-01,10\n2024-01-02,20\n", encoding="utf-8")

    cfg = Path("clients/example_client/config.yml")
    df = run_pipeline(input_path=p, client_config_path=cfg)

    assert set(["date", "value"]).issubset(df.columns)
    assert df["value"].sum() == 30
