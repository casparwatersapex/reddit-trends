from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.embedding_benchmark import build_benchmark_rows, load_env


def test_build_benchmark_rows_estimates_cost_and_time() -> None:
    df = pd.DataFrame(
        {
            "title": ["One", "Two"],
            "body": ["Alpha", "Beta"],
        }
    )
    rows = build_benchmark_rows(
        df=df,
        title_col="title",
        body_col="body",
        sample_rows=2,
        seed=7,
        local_model="local-test",
        local_rows_per_sec=10.0,
        openai_model="openai-test",
        openai_tokens_per_sec=1000.0,
        openai_cost_per_million=1.0,
        openai_live=False,
        openai_api_key=None,
        openai_batch_size=64,
        openai_timeout_s=10,
    )

    assert len(rows) == 2
    local = rows[0]
    openai = rows[1]
    assert local["option"] == "local"
    assert openai["option"] == "openai"
    assert isinstance(openai["est_time_hours"], float)
    assert isinstance(openai["est_cost_usd"], float)


def test_load_env_sets_missing_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=abc123\n", encoding="utf-8")
    os.environ.pop("OPENAI_API_KEY", None)

    load_env(env_path)

    assert os.environ["OPENAI_API_KEY"] == "abc123"
