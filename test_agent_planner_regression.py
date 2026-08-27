from __future__ import annotations

import pandas as pd
import pytest

from analyst import deterministic_plan


@pytest.fixture
def planner_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": ["North", "South", "North", "East"],
            "Product": ["Laptop", "Phone", "Laptop", "Tablet"],
            "Sales": [100, 200, 150, 50],
            "Date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-15",
                    "2025-02-01",
                    "2025-02-15",
                ]
            ),
        }
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the total Sales?",
            {
                "operation": "calculate_sum",
                "column": "Sales",
            },
        ),
        (
            "What is the average Sales?",
            {
                "operation": "calculate_average",
                "column": "Sales",
            },
        ),
        (
            "How many records are there?",
            {
                "operation": "calculate_count",
                "column": "Region",
            },
        ),
        (
            "How many unique Products are there?",
            {
                "operation": "calculate_unique_count",
                "column": "Product",
            },
        ),
        (
            "Show Sales by Region",
            {
                "operation": "group_and_sum",
                "group_column": "Region",
                "value_column": "Sales",
            },
        ),
        (
            "Show average Sales by Region",
            {
                "operation": "group_and_average",
                "group_column": "Region",
                "value_column": "Sales",
            },
        ),
        (
            "Show the top 2 Regions by Sales",
            {
                "operation": "top_n",
                "group_column": "Region",
                "value_column": "Sales",
                "n": 2,
            },
        ),
        (
            "Show Sales in North",
            {
                "operation": "filtered_sum",
                "filters": [
                    {
                        "column": "Region",
                        "operator": "=",
                        "value": "North",
                    }
                ],
                "value_column": "Sales",
            },
        ),
        (
            "Show Sales in North for Laptop",
            {
                "operation": "filtered_sum",
                "filters": [
                    {
                        "column": "Region",
                        "operator": "=",
                        "value": "North",
                    },
                    {
                        "column": "Product",
                        "operator": "=",
                        "value": "Laptop",
                    },
                ],
                "value_column": "Sales",
            },
        ),
        (
            "Show monthly Sales",
            {
                "operation": "monthly_sum",
                "date_column": "Date",
                "value_column": "Sales",
            },
        ),
    ],
)
def test_deterministic_planner_regression(
    planner_df: pd.DataFrame,
    question: str,
    expected: dict,
) -> None:
    """Lock down representative natural-language planner behavior."""
    assert deterministic_plan(question, planner_df) == expected
