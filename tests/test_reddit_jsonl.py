from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from client_portal.pipeline.run import run_pipeline


def test_reddit_jsonl_pipeline(tmp_path: Path) -> None:
    p = tmp_path / "input.jsonl"
    rows = [
        {
            "id": "abc123",
            "created_utc": 1_700_000_000,
            "subreddit": "GardeningUK",
            "title": "Sample title",
            "selftext": "Sample body",
            "url": "https://example.com/post",
            "author": "user1",
            "score": 5,
            "num_comments": 2,
            "is_self": True,
            "permalink": "/r/GardeningUK/comments/abc123/sample_title/",
        }
    ]
    p.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cfg = Path("clients/gardeninguk/config.yml")
    df = run_pipeline(input_path=p, client_config_path=cfg)

    required = {
        "post_id",
        "date",
        "subreddit",
        "title",
        "body",
        "url",
        "author",
        "score",
        "num_comments",
        "is_self",
        "permalink",
    }
    assert required.issubset(df.columns)
    assert df.loc[0, "post_id"] == "abc123"
    assert df.loc[0, "body"] == "Sample body"
    assert df.loc[0, "date"] == pd.to_datetime(1_700_000_000, unit="s")
