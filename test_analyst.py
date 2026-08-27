import json

import pandas as pd
import pytest

import analyst


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "Product": [
                "Laptop",
                "Phone",
                "Tablet",
                "Laptop",
            ],
            "Revenue": [
                50000,
                30000,
                20000,
                40000,
            ],
            "Units": [
                10,
                20,
                15,
                8,
            ],
            "City": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Mumbai",
            ],
        }
    )


@pytest.fixture
def sample_profile(sample_dataframe):
    return analyst.build_profile(sample_dataframe)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        if not self.responses:
            raise AssertionError(
                "Unexpected Gemini call."
            )

        return FakeResponse(
            self.responses.pop(0)
        )


class FakeGeminiClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


# ============================================================
# GEMINI CLIENT
# ============================================================


def test_get_gemini_client_requires_api_key(
    monkeypatch,
):
    monkeypatch.setattr(
        analyst,
        "GEMINI_API_KEY",
        None,
    )

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY is missing",
    ):
        analyst.get_gemini_client()


# ============================================================
# GEMINI PLANNER
# ============================================================


def test_gemini_plan_accepts_valid_json(
    sample_profile,
    monkeypatch,
):
    response = json.dumps(
        {
            "operation": "calculate_sum",
            "column": "Revenue",
        }
    )

    client = FakeGeminiClient(
        [response]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    plan = analyst._gemini_plan(
        "What is the total revenue?",
        sample_profile,
    )

    assert plan["operation"] == "calculate_sum"
    assert plan["column"] == "Revenue"

    assert len(
        client.models.calls
    ) == 1

    call = client.models.calls[0]

    assert call["model"] == analyst.MODEL_NAME
    assert call["config"][
        "response_mime_type"
    ] == "application/json"


def test_gemini_plan_rejects_empty_response(
    sample_profile,
    monkeypatch,
):
    client = FakeGeminiClient(
        [""]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty analysis plan",
    ):
        analyst._gemini_plan(
            "What is the total revenue?",
            sample_profile,
        )


# ============================================================
# CHOOSE ANALYSIS
# ============================================================


def test_choose_analysis_accepts_valid_gemini_plan(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    response = json.dumps(
        {
            "operation": "calculate_sum",
            "column": "Revenue",
        }
    )

    client = FakeGeminiClient(
        [response]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    plan = analyst.choose_analysis(
        "What is the total revenue?",
        sample_profile,
        df=sample_dataframe,
    )

    assert plan["operation"] == "calculate_sum"
    assert plan["column"] == "Revenue"


def test_choose_analysis_requires_dataframe(
    sample_profile,
):
    with pytest.raises(
        ValueError,
        match="requires the real DataFrame",
    ):
        analyst.choose_analysis(
            "What is the total revenue?",
            sample_profile,
        )


def test_choose_analysis_rejects_invalid_column(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    response = json.dumps(
        {
            "operation": "calculate_sum",
            "column": "NotARealColumn",
        }
    )

    client = FakeGeminiClient(
        [response]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini could not create a valid analysis plan",
    ):
        analyst.choose_analysis(
            "What is the total revenue?",
            sample_profile,
            df=sample_dataframe,
        )


def test_choose_analysis_rejects_invalid_operation(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    response = json.dumps(
        {
            "operation": "invented_operation",
            "column": "Revenue",
        }
    )

    client = FakeGeminiClient(
        [response]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini could not create a valid analysis plan",
    ):
        analyst.choose_analysis(
            "What is the total revenue?",
            sample_profile,
            df=sample_dataframe,
        )


def test_choose_analysis_rejects_invalid_filter_column(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    response = json.dumps(
        {
            "operation": "filtered_sum",
            "value_column": "Revenue",
            "filters": [
                {
                    "column": "NotARealColumn",
                    "operator": "=",
                    "value": "Delhi",
                }
            ],
        }
    )

    client = FakeGeminiClient(
        [response]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini could not create a valid analysis plan",
    ):
        analyst.choose_analysis(
            "What is the total revenue in Delhi?",
            sample_profile,
            df=sample_dataframe,
        )


# ============================================================
# EXPLANATION
# ============================================================


def test_explain_result_returns_gemini_text(
    monkeypatch,
):
    client = FakeGeminiClient(
        [
            "The total revenue is 140,000."
        ]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    result = analyst.explain_result(
        "What is the total revenue?",
        {
            "operation": "calculate_sum",
            "column": "Revenue",
        },
        140000.0,
    )

    assert result == (
        "The total revenue is 140,000."
    )

    assert len(
        client.models.calls
    ) == 1


def test_explain_result_rejects_empty_response(
    monkeypatch,
):
    client = FakeGeminiClient(
        [""]
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini returned an empty explanation",
    ):
        analyst.explain_result(
            "What is the total revenue?",
            {
                "operation": "calculate_sum",
                "column": "Revenue",
            },
            140000.0,
        )


def test_explain_result_wraps_gemini_failure(
    monkeypatch,
):
    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError(
                "API unavailable"
            )

    class FailingClient:
        models = FailingModels()

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: FailingClient(),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini failed while explaining",
    ):
        analyst.explain_result(
            "What is the total revenue?",
            {
                "operation": "calculate_sum",
                "column": "Revenue",
            },
            140000.0,
        )


# ============================================================
# COMPLETE PIPELINE
# ============================================================


def test_run_analysis_executes_python_result_then_explains(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    responses = [
        json.dumps(
            {
                "operation": "calculate_sum",
                "column": "Revenue",
            }
        ),
        "The total revenue is 140,000.",
    ]

    client = FakeGeminiClient(
        responses
    )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    output = analyst.run_analysis(
        sample_dataframe,
        sample_profile,
        "What is the total revenue?",
    )

    assert output["plan"]["operation"] == (
        "calculate_sum"
    )

    assert output["plan"]["column"] == (
        "Revenue"
    )

    assert output["result"] == 140000.0

    assert output["explanation"] == (
        "The total revenue is 140,000."
    )

    assert len(
        client.models.calls
    ) == 2


def test_run_analysis_uses_actual_python_result(
    sample_dataframe,
    sample_profile,
    monkeypatch,
):
    captured_explanation_call = {}

    planner_response = json.dumps(
        {
            "operation": "filtered_sum",
            "value_column": "Revenue",
            "filters": [
                {
                    "column": "City",
                    "operator": "=",
                    "value": "Delhi",
                }
            ],
        }
    )

    class TrackingModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)

            if len(self.calls) == 1:
                return FakeResponse(
                    planner_response
                )

            captured_explanation_call.update(
                kwargs
            )

            return FakeResponse(
                "Delhi revenue is 70,000."
            )

    class TrackingClient:
        def __init__(self):
            self.models = TrackingModels()

    client = TrackingClient()

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        lambda: client,
    )

    output = analyst.run_analysis(
        sample_dataframe,
        sample_profile,
        "What is the total revenue in Delhi?",
    )

    assert output["result"] == 70000.0

    assert output["explanation"] == (
        "Delhi revenue is 70,000."
    )

    explanation_prompt = (
        captured_explanation_call["contents"]
    )

    assert "70000.0" in explanation_prompt

    assert "What is the total revenue in Delhi?" in (
        explanation_prompt
    )


def test_run_analysis_rejects_empty_question(
    sample_dataframe,
    sample_profile,
):
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        analyst.run_analysis(
            sample_dataframe,
            sample_profile,
            "",
        )


def test_run_analysis_does_not_call_gemini_for_invalid_dataframe(
    sample_profile,
    monkeypatch,
):
    calls = []

    def fail_if_called():
        calls.append(True)

        raise AssertionError(
            "Gemini should not be called."
        )

    monkeypatch.setattr(
        analyst,
        "get_gemini_client",
        fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match="The dataset is empty",
    ):
        analyst.run_analysis(
            pd.DataFrame(),
            sample_profile,
            "What is the total revenue?",
        )

    assert not calls


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================


def test_execute_plan_remains_backward_compatible(
    sample_dataframe,
):
    plan = {
        "operation": "calculate_sum",
        "column": "Revenue",
    }

    result = analyst.execute_plan(
        sample_dataframe,
        plan,
    )

    assert result == 140000.0

