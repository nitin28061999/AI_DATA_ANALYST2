import pandas as pd
import pytest

from data_profile import (
    _classify_column,
    _get_sample_values,
    _is_id_like,
    _numeric_summary,
    create_profile,
    get_profile,
    profile_column,
)


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer_ID": [101, 102, 103, 104],
            "Region": ["North", "South", "North", "East"],
            "Sales": [100.0, 200.0, 150.0, 300.0],
            "Order_Date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "Is_Active": [True, False, True, True],
            "Description": [
                "First order",
                "Second order",
                "Third order",
                "Fourth order",
            ],
        }
    )


# ============================================================
# ID DETECTION
# ============================================================

def test_id_like_column_by_name(sample_df):
    assert _is_id_like(
        "Customer_ID",
        sample_df["Customer_ID"],
    ) is True


def test_id_like_high_cardinality_text():
    series = pd.Series(
        [
            "A001",
            "A002",
            "A003",
            "A004",
            "A005",
        ]
    )

    assert _is_id_like(
        "Reference",
        series,
    ) is True


def test_non_id_column(sample_df):
    assert _is_id_like(
        "Region",
        sample_df["Region"],
    ) is False


# ============================================================
# COLUMN CLASSIFICATION
# ============================================================

def test_numeric_column_classification(sample_df):
    assert _classify_column(
        "Sales",
        sample_df["Sales"],
    ) == "numeric"


def test_id_column_classification(sample_df):
    assert _classify_column(
        "Customer_ID",
        sample_df["Customer_ID"],
    ) == "id"


def test_categorical_column_classification(sample_df):
    assert _classify_column(
        "Region",
        sample_df["Region"],
    ) == "categorical"


def test_datetime_column_classification(sample_df):
    assert _classify_column(
        "Order_Date",
        sample_df["Order_Date"],
    ) == "datetime"


def test_boolean_column_classification(sample_df):
    assert _classify_column(
        "Is_Active",
        sample_df["Is_Active"],
    ) == "boolean"


def test_text_column_classification():
    series = pd.Series(
        [
            "This is a long description.",
            "Another unique description.",
            "A third unique description.",
            "A fourth unique description.",
        ]
    )

    result = _classify_column(
        "Description",
        series,
    )

    assert result in {"categorical", "text"}


# ============================================================
# STRING DATE DETECTION
# ============================================================

def test_string_dates_are_detected_as_datetime():
    series = pd.Series(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
        ]
    )

    assert _classify_column(
        "Date",
        series,
    ) == "datetime"


# ============================================================
# SAMPLE VALUES
# ============================================================

def test_get_sample_values(sample_df):
    values = _get_sample_values(
        sample_df["Region"]
    )

    assert values == [
        "North",
        "South",
        "East",
    ]


def test_get_sample_values_removes_nulls():
    series = pd.Series(
        [
            "A",
            None,
            "B",
            "A",
            None,
            "C",
        ]
    )

    values = _get_sample_values(series)

    assert values == [
        "A",
        "B",
        "C",
    ]


def test_get_sample_values_respects_limit():
    series = pd.Series(
        ["A", "B", "C", "D", "E"]
    )

    values = _get_sample_values(
        series,
        limit=3,
    )

    assert len(values) == 3
    assert values == [
        "A",
        "B",
        "C",
    ]


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def test_numeric_summary():
    series = pd.Series(
        [100, 200, 300, 400]
    )

    summary = _numeric_summary(series)

    assert summary["min"] == 100.0
    assert summary["max"] == 400.0
    assert summary["mean"] == 250.0
    assert summary["median"] == 250.0


def test_numeric_summary_ignores_invalid_values():
    series = pd.Series(
        [
            100,
            "200",
            "invalid",
            None,
            300,
        ]
    )

    summary = _numeric_summary(series)

    assert summary["min"] == 100.0
    assert summary["max"] == 300.0
    assert summary["mean"] == 200.0
    assert summary["median"] == 200.0


def test_numeric_summary_empty_result():
    series = pd.Series(
        [
            "abc",
            "xyz",
            None,
        ]
    )

    summary = _numeric_summary(series)

    assert summary == {}


# ============================================================
# COLUMN PROFILE
# ============================================================

def test_profile_numeric_column(sample_df):
    profile = profile_column(
        "Sales",
        sample_df["Sales"],
    )

    assert profile["name"] == "Sales"
    assert profile["dtype"] == str(
        sample_df["Sales"].dtype
    )
    assert profile["kind"] == "numeric"
    assert profile["row_count"] == 4
    assert profile["missing_count"] == 0
    assert profile["missing_percentage"] == 0.0
    assert profile["unique_count"] == 4

    assert "numeric_summary" in profile

    assert profile["numeric_summary"]["min"] == 100.0
    assert profile["numeric_summary"]["max"] == 300.0
    assert profile["numeric_summary"]["mean"] == 187.5
    assert profile["numeric_summary"]["median"] == 175.0


def test_profile_categorical_column(sample_df):
    profile = profile_column(
        "Region",
        sample_df["Region"],
    )

    assert profile["name"] == "Region"
    assert profile["kind"] == "categorical"
    assert profile["row_count"] == 4
    assert profile["missing_count"] == 0
    assert profile["unique_count"] == 3

    assert "numeric_summary" not in profile


# ============================================================
# DATASET PROFILE
# ============================================================

def test_create_profile_structure(sample_df):
    profile = create_profile(sample_df)

    assert isinstance(profile, dict)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 6

    assert "columns" in profile
    assert "column_names" in profile

    assert "numeric_columns" in profile
    assert "categorical_columns" in profile
    assert "datetime_columns" in profile
    assert "id_columns" in profile
    assert "text_columns" in profile
    assert "boolean_columns" in profile

    assert "total_missing_values" in profile
    assert "duplicate_rows" in profile


def test_create_profile_column_names(sample_df):
    profile = create_profile(sample_df)

    assert profile["column_names"] == [
        "Customer_ID",
        "Region",
        "Sales",
        "Order_Date",
        "Is_Active",
        "Description",
    ]


def test_create_profile_column_categories(sample_df):
    profile = create_profile(sample_df)

    assert "Sales" in profile["numeric_columns"]
    assert "Region" in profile["categorical_columns"]
    assert "Order_Date" in profile["datetime_columns"]
    assert "Customer_ID" in profile["id_columns"]
    assert "Is_Active" in profile["boolean_columns"]


# ============================================================
# MISSING VALUES
# ============================================================

def test_profile_missing_values():
    df = pd.DataFrame(
        {
            "Name": [
                "A",
                "B",
                None,
                "D",
            ],
            "Sales": [
                100,
                None,
                300,
                None,
            ],
        }
    )

    profile = create_profile(df)

    assert profile["row_count"] == 4
    assert profile["total_missing_values"] == 3


def test_profile_missing_percentage():
    df = pd.DataFrame(
        {
            "Sales": [
                100,
                200,
                None,
                None,
            ]
        }
    )

    profile = create_profile(df)

    sales_profile = next(
        column
        for column in profile["columns"]
        if column["name"] == "Sales"
    )

    assert sales_profile["missing_count"] == 2
    assert sales_profile["missing_percentage"] == 50.0


# ============================================================
# DUPLICATE ROWS
# ============================================================

def test_duplicate_rows():
    df = pd.DataFrame(
        {
            "Region": [
                "North",
                "North",
                "South",
            ],
            "Sales": [
                100,
                100,
                200,
            ],
        }
    )

    profile = create_profile(df)

    assert profile["duplicate_rows"] == 1


# ============================================================
# INVALID INPUT
# ============================================================

def test_create_profile_requires_dataframe():
    with pytest.raises(
        TypeError,
        match="df must be a pandas DataFrame",
    ):
        create_profile(
            [
                {"Sales": 100}
            ] # pyright: ignore[reportArgumentType]
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def test_get_profile_matches_create_profile(sample_df):
    profile_from_create = create_profile(
        sample_df
    )

    profile_from_get = get_profile(
        sample_df
    )

    assert profile_from_get == profile_from_create