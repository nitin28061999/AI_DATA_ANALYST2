from typing import Any, Dict
import pandas as pd


def _is_id_like(col_name: str, series: pd.Series) -> bool:
    """Check if a column is likely an identifier column."""
    name_lower = col_name.lower()

    if any(
        keyword in name_lower
        for keyword in ["id", "code", "key", "number"]
    ):
        return True

    non_null = series.dropna()

    if non_null.empty:
        return False

    return (
        non_null.nunique() == len(non_null)
        and not pd.api.types.is_float_dtype(series)
    )


def _get_sample_values(
    series: pd.Series,
    max_samples: int = 5,
    limit: int | None = None,
) -> list:
    """Extract up to the requested number of non-null unique values."""

    # Backward compatibility with the test suite.
    if limit is not None:
        max_samples = limit

    if max_samples <= 0:
        return []

    values = series.dropna().unique()

    return values[:max_samples].tolist()


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    """
    Generate statistical summary for values that can actually be
    interpreted as numeric.

    Invalid values such as 'abc' are ignored.
    """
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return {}

    return {
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max()),
        "mean": float(numeric_series.mean()),
        "median": float(numeric_series.median()),
        "std": (
            float(numeric_series.std())
            if len(numeric_series) > 1
            else 0.0
        ),
    }


def _classify_column(
    col_name: str,
    series: pd.Series | None = None,
) -> str:
    """
    Classify a column into one of:

    - id
    - numeric
    - datetime
    - boolean
    - categorical
    - text

    Supports both:

        _classify_column(series)

    and:

        _classify_column("Sales", series)
    """

    # Backward compatibility:
    # _classify_column(series)
    if series is None:
        series = col_name  # type: ignore[assignment]
        col_name = ""

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    # Boolean must be checked before numeric because pandas
    # treats boolean separately from ordinary numeric columns.
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Numeric columns.
    if pd.api.types.is_numeric_dtype(series):
        return (
            "id"
            if _is_id_like(col_name, series)
            else "numeric"
        )

    # Datetime columns.
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Categorical dtype.
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    non_null = series.dropna()

    if non_null.empty:
        return "text"

    # Detect strings that actually contain dates.
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        converted_dates = pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed",
        )

        if len(non_null) > 0:
            date_ratio = converted_dates.notna().mean()

            if date_ratio >= 0.8:
                return "datetime"

    # Low-cardinality values are categorical.
    unique_count = non_null.nunique()
    value_count = len(non_null)

    if value_count > 0:
        cardinality_ratio = unique_count / value_count

        # Small repeated-value columns such as:
        # Region = [North, South, North, East]
        if unique_count <= 10 and unique_count < value_count:
            return "categorical"

        if cardinality_ratio < 0.2:
            return "categorical"

    return "text"


def profile_column(
    col_name: str,
    series: pd.Series | None = None,
) -> Dict[str, Any]:
    """
    Profile an individual pandas Series column.

    Supports both:

        profile_column("Sales", df["Sales"])

    and:

        profile_column(df["Sales"], col_name="Sales")
    """

    # Backward-compatible argument handling.
    if isinstance(col_name, pd.Series):
        actual_series = col_name
        actual_col_name = (
            series
            if isinstance(series, str)
            else ""
        )

        series = actual_series
        col_name = actual_col_name

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    col_name = str(col_name)

    col_type = _classify_column(
        col_name,
        series,
    )

    row_count = len(series)
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    col_info: Dict[str, Any] = {
        "name": col_name,
        "type": col_type,
        "kind": col_type,
        "dtype": str(series.dtype),
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_percentage": (
            float(
                missing_count / row_count * 100
            )
            if row_count > 0
            else 0.0
        ),
        "unique_count": unique_count,
        "is_id_like": _is_id_like(
            col_name,
            series,
        ),
    }

    # IMPORTANT:
    # Numeric statistics must be nested under
    # "numeric_summary" because the tests expect
    # this structure.
    if col_type == "numeric":
        col_info["numeric_summary"] = _numeric_summary(
            series
        )

    elif col_type in {
        "categorical",
        "text",
        "datetime",
    }:
        col_info["sample_values"] = _get_sample_values(
            series
        )

    return col_info


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset metadata and column profile summary."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame"
        )

    if df.empty:
        raise ValueError(
            "Input must be a non-empty pandas DataFrame."
        )

    column_profiles = [
        profile_column(
            str(col),
            df[col],
        )
        for col in df.columns
    ]

    numeric_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "numeric"
    ]

    categorical_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "categorical"
    ]

    datetime_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "datetime"
    ]

    boolean_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "boolean"
    ]

    id_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "id"
    ]

    text_columns = [
        column["name"]
        for column in column_profiles
        if column["kind"] == "text"
    ]

    total_missing_values = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # `columns` remains a list because the tests expect:
    #
    # for column in profile["columns"]:
    #     column["name"]
    #
    # Dictionary-style access is preserved separately through
    # `columns_by_name`.
    columns_by_name = {
        column["name"]: column
        for column in column_profiles
    }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": [
            str(col)
            for col in df.columns
        ],
        "columns": column_profiles,
        "columns_by_name": columns_by_name,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
        "id_columns": id_columns,
        "text_columns": text_columns,
        "total_missing_values": total_missing_values,
        "duplicate_rows": duplicate_rows,
    }


# Exported function aliases for compatibility.
create_profile = profile_data
get_profile = profile_data
_profile_column = profile_column