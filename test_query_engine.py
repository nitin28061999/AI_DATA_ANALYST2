
import pandas as pd
import pytest

from query_engine import ( # pyright: ignore[reportMissingImports]
    execute_query,
    validate_query,
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
# QUERY VALIDATION
# ============================================================


class TestQueryValidation:

    def test_valid_sum_query(self):
        query = {
            "operation": "sum",
            "column": "Sales",
        }

        result = validate_query(query)

        assert result["operation"] == "sum"
        assert result["column"] == "Sales"

    def test_operation_is_case_insensitive(self):
        query = {
            "operation": "SUM",
            "column": "Sales",
        }

        result = validate_query(query)

        assert result["operation"] == "sum"

    def test_missing_query_raises_error(self):
        with pytest.raises(
            ValueError,
            match="Query must be a dictionary",
        ):
            validate_query(None)

    def test_missing_operation_raises_error(self):
        query = {
            "column": "Sales",
        }

        with pytest.raises(
            ValueError,
            match="missing 'operation'",
        ):
            validate_query(query)

    def test_missing_column_for_column_operation_raises_error(self):
        query = {
            "operation": "sum",
        }

        with pytest.raises(
            ValueError,
            match="missing 'column'",
        ):
            validate_query(query)

    def test_unsupported_operation_raises_error(self):
        query = {
            "operation": "median",
            "column": "Sales",
        }

        with pytest.raises(
            ValueError,
            match="Unsupported query operation",
        ):
            validate_query(query)

    def test_group_operation_requires_group_by(self):
        query = {
            "operation": "group_sum",
            "column": "Sales",
        }

        with pytest.raises(
            ValueError,
            match="missing 'group_by'",
        ):
            validate_query(query)

    def test_top_n_requires_n(self):
        query = {
            "operation": "top_n",
            "group_by": "Region",
            "column": "Sales",
        }

        with pytest.raises(
            ValueError,
            match="missing 'n'",
        ):
            validate_query(query)


# ============================================================
# BASIC QUERY OPERATIONS
# ============================================================


class TestBasicQueries:

    def test_sum(self, sample_dataframe):
        self._extracted_from_test_max_2("sum", "Sales", sample_dataframe, 920.0)

    def test_average(self, sample_dataframe):
        query = {
            "operation": "average",
            "column": "Sales",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == pytest.approx(
            920.0 / 6
        )

    def test_count(self, sample_dataframe):
        query = {
            "operation": "count",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 6

    def test_count_with_column(self, sample_dataframe):
        self._extracted_from_test_max_2("count", "Sales", sample_dataframe, 6)

    def test_unique_count(self, sample_dataframe):
        self._extracted_from_test_max_2("unique_count", "Region", sample_dataframe, 3)

    def test_min(self, sample_dataframe):
        self._extracted_from_test_max_2("min", "Sales", sample_dataframe, 50.0)

    def test_max(self, sample_dataframe):
        self._extracted_from_test_max_2("max", "Sales", sample_dataframe, 300.0)

    # TODO Rename this here and in `test_sum`, `test_count_with_column`, `test_unique_count`, `test_min` and `test_max`
    def _extracted_from_test_max_2(self, arg0, arg1, sample_dataframe, arg3):
        query = {"operation": arg0, "column": arg1}
        result = execute_query(sample_dataframe, query)
        assert result == arg3


# ============================================================
# GROUPED QUERY OPERATIONS
# ============================================================


class TestGroupedQueries:

    def test_group_sum(self, sample_dataframe):
        query = {
            "operation": "group_sum",
            "group_by": "Region",
            "column": "Sales",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 600.0
        assert result["South"] == 270.0
        assert result["East"] == 50.0

    def test_group_average(self, sample_dataframe):
        query = {
            "operation": "group_average",
            "group_by": "Region",
            "column": "Sales",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 200.0
        assert result["South"] == 135.0
        assert result["East"] == 50.0

    def test_group_count(self, sample_dataframe):
        query = {
            "operation": "group_count",
            "group_by": "Region",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        result = result.set_index(
            "Region"
        )["Count"]

        assert result["North"] == 3
        assert result["South"] == 2
        assert result["East"] == 1

    def test_value_counts(self, sample_dataframe):
        query = {
            "operation": "value_counts",
            "column": "Region",
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        result = result.set_index(
            "Region"
        )["Count"]

        assert result["North"] == 3
        assert result["South"] == 2
        assert result["East"] == 1


# ============================================================
# TOP N
# ============================================================


class TestTopN:

    def test_top_n(self, sample_dataframe):
        query = {
            "operation": "top_n",
            "group_by": "Region",
            "column": "Sales",
            "n": 2,
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result["Region"].tolist() == [
            "North",
            "South",
        ]

        assert result["Sales"].tolist() == [
            600.0,
            270.0,
        ]

    def test_top_n_returns_requested_number(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "top_n",
            "group_by": "Region",
            "column": "Sales",
            "n": 1,
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert len(result) == 1
        assert result["Region"].tolist() == [
            "North",
        ]


# ============================================================
# FILTERED QUERIES
# ============================================================


class TestFilteredQueries:

    def test_filtered_sum(self, sample_dataframe):
        query = {
            "operation": "sum",
            "column": "Sales",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 600.0

    def test_filtered_average(self, sample_dataframe):
        query = {
            "operation": "average",
            "column": "Sales",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 200.0

    def test_filtered_count(self, sample_dataframe):
        query = {
            "operation": "count",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 3

    def test_filtered_unique_count(self, sample_dataframe):
        query = {
            "operation": "unique_count",
            "column": "Region",
            "filters": [
                {
                    "column": "Category",
                    "operator": "=",
                    "value": "A",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 2

    def test_filtered_min(self, sample_dataframe):
        query = {
            "operation": "min",
            "column": "Sales",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 100.0

    def test_filtered_max(self, sample_dataframe):
        query = {
            "operation": "max",
            "column": "Sales",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 300.0


# ============================================================
# MULTIPLE FILTERS
# ============================================================


class TestMultipleFilters:

    def test_multiple_filters_use_and_logic(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "sum",
            "column": "Sales",
            "filters": [
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
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 300.0

    def test_multiple_filters_with_comparison(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "sum",
            "column": "Sales",
            "filters": [
                {
                    "column": "Region",
                    "operator": "=",
                    "value": "North",
                },
                {
                    "column": "Sales",
                    "operator": ">",
                    "value": 150,
                },
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result == 500.0


# ============================================================
# FILTERED GROUP OPERATIONS
# ============================================================


class TestFilteredGroupedQueries:

    def test_filtered_group_sum(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "group_sum",
            "group_by": "Region",
            "column": "Sales",
            "filters": [
                {
                    "column": "Category",
                    "operator": "=",
                    "value": "A",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        result = result.set_index(
            "Region"
        )["Sales"]

        assert result["North"] == 300.0
        assert result["South"] == 120.0

    def test_filtered_group_average(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "group_average",
            "group_by": "Region",
            "column": "Sales",
            "filters": [
                {
                    "column": "Category",
                    "operator": "=",
                    "value": "A",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
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
        query = {
            "operation": "value_counts",
            "column": "Category",
            "filters": [
                {
                    "column": "Region",
                    "operator": "!=",
                    "value": "East",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
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
        query = {
            "operation": "top_n",
            "group_by": "Region",
            "column": "Sales",
            "n": 2,
            "filters": [
                {
                    "column": "Category",
                    "operator": "=",
                    "value": "A",
                }
            ],
        }

        result = execute_query(
            sample_dataframe,
            query,
        )

        assert result["Region"].tolist() == [
            "North",
            "South",
        ]

        assert result["Sales"].tolist() == [
            300.0,
            120.0,
        ]


# ============================================================
# INVALID COLUMN / INVALID DATAFRAME
# ============================================================


class TestQueryErrors:

    def test_invalid_column_raises_error(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "sum",
            "column": "DoesNotExist",
        }

        with pytest.raises(
            (ValueError, KeyError),
        ):
            execute_query(
                sample_dataframe,
                query,
            )

    def test_invalid_group_column_raises_error(
        self,
        sample_dataframe,
    ):
        query = {
            "operation": "group_sum",
            "group_by": "DoesNotExist",
            "column": "Sales",
        }

        with pytest.raises(
            (ValueError, KeyError),
        ):
            execute_query(
                sample_dataframe,
                query,
            )

    def test_empty_dataframe_raises_error(self):
        query = {
            "operation": "sum",
            "column": "Sales",
        }

        with pytest.raises(
            ValueError,
        ):
            execute_query(
                pd.DataFrame(),
                query,
            )

