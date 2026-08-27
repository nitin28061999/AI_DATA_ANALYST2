import pandas as pd
import pytest

from analysis_tools import (
    apply_filters,
    calculate_sum,
    calculate_average,
    calculate_count,
    calculate_unique_count,
    calculate_min,
    calculate_max,
    group_and_sum,
    group_and_average,
    group_and_count,
    top_n,
    value_counts,
    filtered_sum,
    filtered_average,
    filtered_count,
    filtered_unique_count,
    filtered_min,
    filtered_max,
    filtered_group_and_sum,
    filtered_group_and_average,
    filtered_value_counts,
    filtered_top_n,
    normalize_filters,
    validate_dataframe,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": [
                "North",
                "South",
                "North",
                "East",
                "North",
                "South",
            ],
            "Category": [
                "A",
                "B",
                "A",
                "B",
                "B",
                "A",
            ],
            "Sales": [
                100.0,
                150.0,
                200.0,
                50.0,
                300.0,
                120.0,
            ],
        }
    )


# ============================================================
# DATAFRAME VALIDATION
# ============================================================


class TestDataFrameValidation:

    def test_validate_dataframe_valid(
        self,
        sample_dataframe,
    ):
        assert validate_dataframe(
            sample_dataframe
        ) is None

    def test_validate_dataframe_empty(self):
        with pytest.raises(
            ValueError,
            match="The dataset is empty",
        ):
            validate_dataframe(
                pd.DataFrame()
            )

    def test_validate_dataframe_invalid_type(self):
        with pytest.raises(
            TypeError,
            match="df must be a pandas DataFrame",
        ):
            validate_dataframe(
                [1, 2, 3]  # pyright: ignore[reportArgumentType]
            )


# ============================================================
# FILTER NORMALIZATION
# ============================================================


class TestFilterNormalization:

    def test_equals_operator(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "==",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "="

    def test_eq_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "eq",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "="

    def test_greater_than_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "greater than",
                "value": 150,
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == ">"

    def test_greater_than_or_equal_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "gte",
                "value": 150,
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == ">="

    def test_less_than_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "lt",
                "value": 150,
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "<"

    def test_less_than_or_equal_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "lte",
                "value": 150,
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "<="

    def test_not_equal_alias(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "ne",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "!="

    def test_repeated_equals_are_normalized(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "====",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "="

    def test_missing_operator_defaults_to_equals(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result[0]["operator"] == "="

    def test_invalid_column_is_skipped(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "DoesNotExist",
                "operator": "=",
                "value": "North",
            }
        ]

        result = normalize_filters(
            sample_dataframe,
            filters,
        )

        assert result == []

    def test_invalid_operator_raises_error(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "unsupported_operator",
                "value": "North",
            }
        ]

        with pytest.raises(
            ValueError,
            match="Unsupported filter operator",
        ):
            normalize_filters(
                sample_dataframe,
                filters,
            )


# ============================================================
# FILTER ENGINE
# ============================================================


class TestFilterEngine:

    def test_apply_filters_equality(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert len(result) == 3

        assert result["Sales"].tolist() == [
            100.0,
            200.0,
            300.0,
        ]

    def test_apply_filters_not_equal(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "!=",
                "value": "North",
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert len(result) == 3

    def test_apply_filters_greater_than(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": ">",
                "value": 150,
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert result["Sales"].tolist() == [
            200.0,
            300.0,
        ]

    def test_apply_filters_greater_than_or_equal(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": ">=",
                "value": 200,
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert result["Sales"].tolist() == [
            200.0,
            300.0,
        ]

    def test_apply_filters_less_than(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "<",
                "value": 150,
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert result["Sales"].tolist() == [
            100.0,
            50.0,
            120.0,
        ]

    def test_apply_filters_less_than_or_equal(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Sales",
                "operator": "<=",
                "value": 150,
            }
        ]

        result = apply_filters(
            sample_dataframe,
            filters,
        )

        assert result["Sales"].tolist() == [
            100.0,
            150.0,
            50.0,
            120.0,
        ]

    def test_multiple_filters_use_and_logic(
        self,
        sample_dataframe,
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
            sample_dataframe,
            filters,
        )

        assert len(result) == 2

        assert result["Sales"].tolist() == [
            100.0,
            200.0,
        ]


# ============================================================
# BASIC AGGREGATIONS
# ============================================================


class TestAggregations:

    def test_calculate_sum(
        self,
        sample_dataframe,
    ):
        result = calculate_sum(
            sample_dataframe,
            "Sales",
        )

        assert result == 920.0

    def test_calculate_average(
        self,
        sample_dataframe,
    ):
        result = calculate_average(
            sample_dataframe,
            "Sales",
        )

        assert result == pytest.approx(
            920.0 / 6
        )

    def test_calculate_count(
        self,
        sample_dataframe,
    ):
        result = calculate_count(
            sample_dataframe,
        )

        assert result == 6

    def test_calculate_count_with_column(
        self,
        sample_dataframe,
    ):
        result = calculate_count(
            sample_dataframe,
            "Sales",
        )

        assert result == 6

    def test_calculate_unique_count(
        self,
        sample_dataframe,
    ):
        result = calculate_unique_count(
            sample_dataframe,
            "Region",
        )

        assert result == 3

    def test_calculate_min(
        self,
        sample_dataframe,
    ):
        result = calculate_min(
            sample_dataframe,
            "Sales",
        )

        assert result == 50.0

    def test_calculate_max(
        self,
        sample_dataframe,
    ):
        result = calculate_max(
            sample_dataframe,
            "Sales",
        )

        assert result == 300.0


# ============================================================
# GROUPED OPERATIONS
# ============================================================


class TestGroupedOperations:

    def test_group_and_sum(
        self,
        sample_dataframe,
    ):
        result = group_and_sum(
            sample_dataframe,
            "Region",
            "Sales",
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 600.0
        assert result["South"] == 270.0
        assert result["East"] == 50.0

    def test_group_and_average(
        self,
        sample_dataframe,
    ):
        result = group_and_average(
            sample_dataframe,
            "Region",
            "Sales",
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 200.0
        assert result["South"] == 135.0
        assert result["East"] == 50.0

    def test_group_and_count(
        self,
        sample_dataframe,
    ):
        result = group_and_count(
            sample_dataframe,
            "Region",
        )

        result = result.set_index(
            "Region"
        )["Count"]

        assert result["North"] == 3
        assert result["South"] == 2
        assert result["East"] == 1

    def test_top_n(
        self,
        sample_dataframe,
    ):
        result = top_n(
            sample_dataframe,
            "Region",
            "Sales",
            2,
        )

        assert result["Region"].tolist() == [
            "North",
            "South",
        ]

        assert result["Sales"].tolist() == [
            600.0,
            270.0,
        ]

    def test_value_counts(
        self,
        sample_dataframe,
    ):
        result = value_counts(
            sample_dataframe,
            "Region",
        )

        result = result.set_index(
            "Region"
        )["Count"]

        assert result["North"] == 3
        assert result["South"] == 2
        assert result["East"] == 1


# ============================================================
# FILTERED AGGREGATIONS
# ============================================================


class TestFilteredAggregations:

    def test_filtered_sum(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = filtered_sum(
            sample_dataframe,
            filters,
            "Sales",
        )

        assert result == 600.0

    def test_filtered_average(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = filtered_average(
            sample_dataframe,
            filters,
            "Sales",
        )

        assert result == 200.0

    def test_filtered_count(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = filtered_count(
            sample_dataframe,
            filters,
        )

        assert result == 3

    def test_filtered_unique_count(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Category",
                "operator": "=",
                "value": "A",
            }
        ]

        result = filtered_unique_count(
            sample_dataframe,
            filters,
            "Region",
        )

        assert result == 2

    def test_filtered_min(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = filtered_min(
            sample_dataframe,
            filters,
            "Sales",
        )

        assert result == 100.0

    def test_filtered_max(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "=",
                "value": "North",
            }
        ]

        result = filtered_max(
            sample_dataframe,
            filters,
            "Sales",
        )

        assert result == 300.0


# ============================================================
# FILTERED GROUP OPERATIONS
# ============================================================


class TestFilteredGroupedOperations:

    def test_filtered_group_and_sum(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Category",
                "operator": "=",
                "value": "A",
            }
        ]

        result = filtered_group_and_sum(
            sample_dataframe,
            filters,
            "Region",
            "Sales",
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 300.0
        assert result["South"] == 120.0

    def test_filtered_group_and_average(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Category",
                "operator": "=",
                "value": "A",
            }
        ]

        result = filtered_group_and_average(
            sample_dataframe,
            filters,
            "Region",
            "Sales",
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 150.0
        assert result["South"] == 120.0

    def test_filtered_value_counts(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Region",
                "operator": "!=",
                "value": "East",
            }
        ]

        result = filtered_value_counts(
            sample_dataframe,
            filters,
            "Category",
        )

        result = result.set_index(
            "Category"
        )["Count"]

        assert result["A"] == 3
        assert result["B"] == 2

    def test_filtered_top_n(
        self,
        sample_dataframe,
    ):
        filters = [
            {
                "column": "Category",
                "operator": "=",
                "value": "A",
            }
        ]

        result = filtered_top_n(
            sample_dataframe,
            filters,
            "Region",
            "Sales",
            2,
        )

        assert result["Region"].tolist() == [
            "North",
            "South",
        ]

        assert result["Sales"].tolist() == [
            300.0,
            120.0,
        ]