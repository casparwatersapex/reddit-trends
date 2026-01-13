from client_portal.analysis.label_review import extract_llm_value


def test_extract_llm_value_strips_json_fence() -> None:
    raw = '```json\n{"label": "Rose Plan Issues", "summary": "Test"}\n```'
    assert extract_llm_value(raw, "label") == "Rose Plan Issues"
    assert extract_llm_value(raw, "summary") == "Test"


def test_extract_llm_value_handles_plain_text() -> None:
    raw = "Simple label"
    assert extract_llm_value(raw, "label") == "Simple label"


def test_extract_llm_value_handles_quoted_json_prefix() -> None:
    raw = '\'\'\'json {"label": "Tools", "summary": "Usage"}'
    assert extract_llm_value(raw, "label") == "Tools"
