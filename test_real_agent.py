import pandas as pd

import analyst
from data_profile import create_profile


def _sample_dataframe():
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


def test_total_revenue_without_gemini(monkeypatch):
    df = _sample_dataframe()
    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    monkeypatch.setattr(
        analyst,
        "choose_analysis",
        lambda question, profile, df=None: plan,
    )

    monkeypatch.setattr(
        analyst,
        "explain_result",
        lambda question, plan, result:
            "Total revenue is 140,000.",
    )

    response = analyst.run_analysis(
        df,
        profile,
        "What is the total revenue?",
    )

    assert response["result"] == 140000.0
    assert response["plan"]["operation"] == "calculate_sum"
    assert response["explanation"] == "Total revenue is 140,000."


def test_filtered_revenue_without_gemini(monkeypatch):
    df = _sample_dataframe()
    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    monkeypatch.setattr(
        analyst,
        "choose_analysis",
        lambda question, profile, df=None: plan,
    )

    monkeypatch.setattr(
        analyst,
        "explain_result",
        lambda question, plan, result:
            "Delhi revenue is 70,000.",
    )

    response = analyst.run_analysis(
        df,
        profile,
        "What is the total revenue for Delhi?",
    )

    assert response["result"] == 70000.0
    assert response["plan"]["operation"] == "filtered_sum"
    assert response["explanation"] == "Delhi revenue is 70,000."