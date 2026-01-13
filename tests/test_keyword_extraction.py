import pandas as pd

from scripts.keyword_extraction import clean_label, parse_keywords, select_examples


def test_clean_label_strips_wrappers() -> None:
    raw = '```json\n"Small garden ideas"\n```'
    assert clean_label(raw) == "Small garden ideas"


def test_parse_keywords_from_dict() -> None:
    payload = {"keywords": ["small garden makeover", "budget garden ideas"]}
    assert parse_keywords(payload) == ["small garden makeover", "budget garden ideas"]


def test_parse_keywords_from_string() -> None:
    payload = "small garden ideas, micro garden ideas, balcony garden"
    assert parse_keywords(payload) == [
        "small garden ideas",
        "micro garden ideas",
        "balcony garden",
    ]


def test_select_examples_uses_score() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "body": ["Body A", "Body B"],
            "score": [5, 10],
        }
    )
    examples = select_examples(df, sample_size=1, max_chars=50)
    assert examples == ["B - Body B"]
