import pytest

from ai_agent import extract_json


def test_extract_json_direct_object():
    payload = '{"operation": "calculate_sum", "column": "Revenue"}'

    result = extract_json(payload)

    assert result == {
        "operation": "calculate_sum",
        "column": "Revenue",
    }


def test_extract_json_from_markdown_fence():
    payload = """```json
{
    "operation": "filtered_sum",
    "value_column": "Revenue"
}
```"""

    result = extract_json(payload)

    assert result["operation"] == "filtered_sum"
    assert result["value_column"] == "Revenue"


def test_extract_json_rejects_invalid_response():
    with pytest.raises(ValueError):
        extract_json("This response contains no JSON.")