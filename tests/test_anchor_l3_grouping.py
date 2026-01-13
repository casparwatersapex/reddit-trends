import pandas as pd

from scripts.anchor_l3_grouping import clean_text, parse_anchors, select_examples


def test_clean_text_strips_wrappers() -> None:
    raw = '```json\n"Small garden ideas"\n```'
    assert clean_text(raw) == "Small garden ideas"


def test_parse_anchors_from_dict() -> None:
    payload = {"anchors": [{"label": "Small gardens", "keywords": ["small garden ideas"]}]}
    anchors = parse_anchors(payload)
    assert len(anchors) == 1
    assert anchors[0]["label"] == "Small gardens"


def test_parse_anchors_from_raw_json() -> None:
    payload = {
        "raw": '```json\n{"anchors": [{"label": "Balcony gardens", "keywords": ["balcony garden ideas"]}]}\n```'
    }
    anchors = parse_anchors(payload)
    assert len(anchors) == 1
    assert anchors[0]["label"] == "Balcony gardens"


def test_select_examples_uses_score() -> None:
    df = pd.DataFrame({"title": ["A", "B"], "body": ["Body A", "Body B"], "score": [1, 10]})
    examples = select_examples(df, sample_size=1, max_chars=50)
    assert examples == ["B - Body B"]
