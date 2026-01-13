from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.name_clusters import build_prompt, parse_sample_sizes, select_examples


def test_parse_sample_sizes() -> None:
    assert parse_sample_sizes("10,30,50") == [10, 30, 50]


def test_build_prompt_includes_topic_id_and_examples() -> None:
    prompt = build_prompt(7, ["Example A", "Example B"])
    assert "Topic id: 7" in prompt
    assert "Example A" in prompt


def test_select_examples_picks_top_similar() -> None:
    df = pd.DataFrame({"topic": [1, 1], "text": ["A", "B"]})
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    examples = select_examples(df, embeddings, topic_id=1, sample_size=1, max_chars=10)
    assert examples == ["A"]
