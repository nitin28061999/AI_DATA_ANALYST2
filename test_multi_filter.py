import pandas as pd
import pytest

from analysis_tools import (
    apply_filters,
    filtered_sum,
    normalize_filters,
)


@pytest.fixture
def multi_filter_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": [
                "North",
                "North",
                "South",
                "East",
            ],
            "Category": [
                "A",
                "B",
                "A",
                "B",
            ],
            "Sales": [
                100.0,
                200.0,
                150.0,
                300.0,
            ],
        }
    )


# ============================================================
# MULTIPLE FILTER TESTS
# ============================================================

def test_multiple_filters_and_logic(
    multi_filter_df,
):
    filters = [
        {
            "column": "Region",
            "operator": "=",
            "value": "North",
        },
        {
            "column": "Category",
            "operator": "=",
            "value": "A",
        },
    ]

    result = apply_filters(
        multi_filter_df,
        filters,
    )

    assert len(result) == 1
    assert result.iloc[0]["Sales"] == 100.0


def test_filtered_sum_with_multiple_filters(
    multi_filter_df,
):
    filters = [
        {
            "column": "Region",
            "operator": "=",
            "value": "North",
        },
        {
            "column": "Sales",
            "operator": ">",
            "value": 150.0,
        },
    ]

    total = filtered_sum(
        multi_filter_df,
        filters,
        "Sales",
    )

    assert total == 200.0


# ============================================================
# OPERATOR NORMALIZATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "operator,expected",
    [
        ("=", "="),
        ("==", "="),
        ("===", "="),
        ("====", "="),
        ("eq", "="),
        ("equals", "="),
        ("equal", "="),
        ("!=", "!="),
        ("ne", "!="),
        ("not equal", "!="),
        ("not_equal", "!="),
        (">", ">"),
        ("gt", ">"),
        ("greater than", ">"),
        ("greater_than", ">"),
        (">=", ">="),
        ("gte", ">="),
        ("greater than or equal", ">="),
        ("greater_than_or_equal", ">="),
        ("<", "<"),
        ("lt", "<"),
        ("less than", "<"),
        ("less_than", "<"),
        ("<=", "<="),
        ("lte", "<="),
        ("less than or equal", "<="),
        ("less_than_or_equal", "<="),
    ],
)
def test_operator_aliases_are_normalized(
    multi_filter_df,
    operator,
    expected,
):
    filters = [
        {
            "column": "Sales",
            "operator": operator,
            "value": 150,
        }
    ]

    normalized = normalize_filters(
        multi_filter_df,
        filters,
    )

    assert len(normalized) == 1
    assert normalized[0]["operator"] == expected


# ============================================================
# OPERATOR EXECUTION TESTS
# ============================================================

def test_equal_alias_filter(
    multi_filter_df,
):
    filters = [
        {
            "column": "Region",
            "operator": "==",
            "value": "North",
        }
    ]

    result = apply_filters(
        multi_filter_df,
        filters,
    )

    assert len(result) == 2


def test_greater_than_or_equal_alias_filter(
    multi_filter_df,
):
    filters = [
        {
            "column": "Sales",
            "operator": "gte",
            "value": 200,
        }
    ]

    result = apply_filters(
        multi_filter_df,
        filters,
    )

    assert len(result) == 2

    assert result["Sales"].tolist() == [
        200.0,
        300.0,
    ]


def test_less_than_alias_filter(
    multi_filter_df,
):
    filters = [
        {
            "column": "Sales",
            "operator": "lt",
            "value": 200,
        }
    ]

    result = apply_filters(
        multi_filter_df,
        filters,
    )

    assert len(result) == 2

    assert result["Sales"].tolist() == [
        100.0,
        150.0,
    ]


# ============================================================
# INVALID FILTER TESTS
# ============================================================

def test_invalid_operator_is_rejected(
    multi_filter_df,
):
    filters = [
        {
            "column": "Sales",
            "operator": "unsupported_operator",
            "value": 100,
        }
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported filter operator",
    ):
        normalize_filters(
            multi_filter_df,
            filters,
        )


def test_unknown_column_is_ignored_by_normalize_filters(
    multi_filter_df,
):
    filters = [
        {
            "column": "UnknownColumn",
            "operator": "=",
            "value": "North",
        }
    ]

    normalized = normalize_filters(
        multi_filter_df,
        filters,
    )

    assert normalized == []