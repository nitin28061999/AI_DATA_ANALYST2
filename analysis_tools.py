from __future__ import annotations

import contextlib
import numbers
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that df is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")


def validate_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a column exists."""
    validate_dataframe(df)

    if not isinstance(column, str) or not column.strip():
        raise ValueError("Column name cannot be empty.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def validate_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a column exists and is numeric."""
    validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )


# ============================================================
# FILTER VALIDATION
# ============================================================

SUPPORTED_OPERATORS = {
    "=",
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "contains",
    "between",
}


def validate_filters(
    df: pd.DataFrame,
    filters: list,
) -> None:
    """
    Validate a list of filters.

    Filters use AND logic.
    Missing operator defaults to "=".
    """
    validate_dataframe(df)

    if not isinstance(filters, list):
        raise ValueError("Filters must be a list.")

    if not filters:
        raise ValueError("At least one filter is required.")

    normalized_operators = {
        _normalize_operator(operator)
        for operator in SUPPORTED_OPERATORS
    }

    for index, filter_item in enumerate(filters):

        if not isinstance(filter_item, dict):
            raise ValueError(
                f"Filter #{index + 1} must be a dictionary."
            )

        if "column" not in filter_item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'column'."
            )

        if "value" not in filter_item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'value'."
            )

        column = filter_item["column"]

        validate_column(
            df,
            column,
        )

        operator = _normalize_operator(
            filter_item.get(
                "operator",
                "=",
            )
        )

        if operator not in normalized_operators:
            raise ValueError(
                f"Unsupported filter operator '{operator}'. "
                f"Supported operators: "
                f"{sorted(SUPPORTED_OPERATORS)}"
            )

        if operator == "between":

            value = filter_item["value"]

            if not isinstance(
                value,
                (list, tuple),
            ) or len(value) != 2:
                raise ValueError(
                    "The 'between' operator requires "
                    "exactly two values."
                )


# ============================================================
# VALUE CONVERSION
# ============================================================

def _normalize_operator(
    operator: Any,
) -> str:
    """Normalize Gemini/user filter operators."""

    if operator is None:
        return "="

    normalized = str(operator).strip().lower()

    # Treat repeated equality characters as equality.
    if normalized and set(normalized) == {"="}:
        return "="

    aliases = {
        "==": "=",
        "eq": "=",
        "equals": "=",
        "equal": "=",

        "not equal": "!=",
        "not_equal": "!=",
        "ne": "!=",

        "greater than": ">",
        "greater_than": ">",
        "gt": ">",

        "greater than or equal": ">=",
        "greater_than_or_equal": ">=",
        "gte": ">=",

        "less than": "<",
        "less_than": "<",
        "lt": "<",

        "less than or equal": "<=",
        "less_than_or_equal": "<=",
        "lte": "<=",

        "is": "=",
        "is equal to": "=",
        "is not": "!=",

        "greater": ">",
        "less": "<",

        "includes": "contains",
        "include": "contains",
        "contains": "contains",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def _convert_value_for_series(
    series: pd.Series,
    value: Any,
) -> Any:
    """
    Convert a filter value to a type compatible with a pandas Series.

    Numeric columns:
        "50000" -> 50000

    Datetime columns:
        "2025-01-01" -> Timestamp

    Text columns:
        value remains unchanged.
    """

    if isinstance(value, (list, tuple)):
        return type(value)(
            _convert_value_for_series(
                series,
                item,
            )
            for item in value
        )

    if pd.api.types.is_numeric_dtype(series):

        try:
            converted = pd.to_numeric(
                value,
                errors="raise",
            )

            return converted.item() if isinstance(converted, np.generic) else converted
        except (
            ValueError,
            TypeError,
        ):
            return value

    if pd.api.types.is_datetime64_any_dtype(series):

        try:
            return pd.to_datetime(
                value,
                errors="raise",
            )

        except (
            ValueError,
            TypeError,
        ):
            return value

    return value


def _is_text_series(
    series: pd.Series,
) -> bool:
    """Return True when a Series should be treated as text."""

    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _text_normalize(
    value: Any,
) -> str:
    """
    Normalize text for human-friendly comparisons.

    Examples:
        " Delhi " -> "delhi"
        "LAPTOP"  -> "laptop"
    """

    return "" if value is None else str(value).strip().casefold()


def _build_text_equality_mask(
    series: pd.Series,
    value: Any,
) -> pd.Series:
    """Build a case-insensitive text equality mask."""

    normalized_value = _text_normalize(value)

    return (
        series
        .map(_text_normalize)
        .eq(normalized_value)
    )


def _build_text_inequality_mask(
    series: pd.Series,
    value: Any,
) -> pd.Series:
    """Build a case-insensitive text inequality mask."""

    normalized_value = _text_normalize(value)

    return (
        series
        .map(_text_normalize)
        .ne(normalized_value)
    )


# ============================================================
# FILTER ENGINE
# ============================================================

def apply_filters(
    df: pd.DataFrame,
    filters: list,
) -> pd.DataFrame:  # sourcery skip: low-code-quality
    """
    Apply multiple filters using AND logic.

    Supported operators:

        =
        ==
        !=
        >
        >=
        <
        <=
        contains
        between

    Filter values are converted to the target column type before
    comparison.
    """

    validate_filters(
        df,
        filters,
    )

    mask = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    for filter_item in filters:

        column = filter_item["column"]

        operator = _normalize_operator(
            filter_item.get(
                "operator",
                "=",
            )
        )

        raw_value = filter_item["value"]

        series = df[column]

        value = _convert_value_for_series(
            series,
            raw_value,
        )

        # ----------------------------------------------------
        # EQUALITY
        # ----------------------------------------------------

        if operator == "=":

            if _is_text_series(series):

                current_mask = (
                    _build_text_equality_mask(
                        series,
                        value,
                    )
                )

            else:

                current_mask = (
                    series == value
                )

            mask &= current_mask

        # ----------------------------------------------------
        # NOT EQUAL
        # ----------------------------------------------------

        elif operator == "!=":

            if _is_text_series(series):

                current_mask = (
                    _build_text_inequality_mask(
                        series,
                        value,
                    )
                )

            else:

                current_mask = (
                    series != value
                )

            mask &= current_mask

        # ----------------------------------------------------
        # GREATER THAN
        # ----------------------------------------------------

        elif operator == ">":

            try:
                current_mask = (
                    series > value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Cannot compare column '{column}' "
                    f"using operator '>'. "
                    f"Value '{raw_value}' is incompatible "
                    f"with the column type."
                ) from exc

            mask &= current_mask

        # ----------------------------------------------------
        # GREATER THAN OR EQUAL
        # ----------------------------------------------------

        elif operator == ">=":

            try:
                current_mask = (
                    series >= value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Cannot compare column '{column}' "
                    f"using operator '>='. "
                    f"Value '{raw_value}' is incompatible "
                    f"with the column type."
                ) from exc

            mask &= current_mask

        # ----------------------------------------------------
        # LESS THAN
        # ----------------------------------------------------

        elif operator == "<":

            try:
                current_mask = (
                    series < value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Cannot compare column '{column}' "
                    f"using operator '<'. "
                    f"Value '{raw_value}' is incompatible "
                    f"with the column type."
                ) from exc

            mask &= current_mask

        # ----------------------------------------------------
        # LESS THAN OR EQUAL
        # ----------------------------------------------------

        elif operator == "<=":

            try:
                current_mask = (
                    series <= value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Cannot compare column '{column}' "
                    f"using operator '<='. "
                    f"Value '{raw_value}' is incompatible "
                    f"with the column type."
                ) from exc

            mask &= current_mask

        # ----------------------------------------------------
        # CONTAINS
        # ----------------------------------------------------

        elif operator == "contains":

            # regex=False is intentional.
            # User/Gemini values should be treated literally.
            current_mask = (
                series
                .astype("string")
                .str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False,
                )
            )

            mask &= current_mask

        # ----------------------------------------------------
        # BETWEEN
        # ----------------------------------------------------

        elif operator == "between":

            if not isinstance(
                value,
                (list, tuple),
            ) or len(value) != 2:

                raise ValueError(
                    "The 'between' filter requires "
                    "exactly two values."
                )

            lower = _convert_value_for_series(
                series,
                value[0],
            )

            upper = _convert_value_for_series(
                series,
                value[1],
            )

            try:
                current_mask = series.between(
                    lower,
                    upper,
                    inclusive="both",
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Cannot apply 'between' filter to "
                    f"column '{column}'."
                ) from exc

            mask &= current_mask

        else:
            raise ValueError(
                f"Unsupported filter operator "
                f"'{operator}'."
            )

    return df.loc[mask].copy()


# ============================================================
# BASIC AGGREGATIONS
# ============================================================

def calculate_sum(
    df: pd.DataFrame,
    column: str,
) -> float:
    """Calculate the sum of a numeric column."""

    validate_numeric_column(
        df,
        column,
    )

    return float(
        df[column].sum()
    )


def calculate_average(
    df: pd.DataFrame,
    column: str,
) -> float:
    """Calculate the average of a numeric column."""

    validate_numeric_column(
        df,
        column,
    )

    values = df[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column '{column}' contains no numeric values."
        )

    return float(
        values.mean()
    )


def calculate_count(
    df: pd.DataFrame,
    column: str | None = None,
) -> int:
    """
    Count rows, or non-null values when a column is supplied.
    """

    validate_dataframe(df)

    if column is None:
        return len(df)

    validate_column(
        df,
        column,
    )

    return int(
        df[column].count()
    )


def calculate_unique_count(
    df: pd.DataFrame,
    column: str,
) -> int:
    """Count unique non-null values."""

    validate_column(
        df,
        column,
    )

    return int(
        df[column].nunique(
            dropna=True
        )
    )


def calculate_min(
    df: pd.DataFrame,
    column: str,
) -> Any:
    """Return the minimum non-null value."""

    validate_column(
        df,
        column,
    )

    values = df[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column '{column}' contains no values."
        )

    return values.min()


def calculate_max(
    df: pd.DataFrame,
    column: str,
) -> Any:
    """Return the maximum non-null value."""

    validate_column(
        df,
        column,
    )

    values = df[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column '{column}' contains no values."
        )

    return values.max()


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_and_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group by a column and calculate sums."""

    validate_column(
        df,
        group_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    if group_column == value_column:

        result = (
            df.groupby(
                group_column,
                dropna=False,
            )
            .agg(
                **{
                    "Sum": (
                        value_column,
                        "sum",
                    )
                }
            )
            .reset_index()
        )

        return (
            result
            .sort_values(
                by="Sum",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    result = (
        df.groupby(
            group_column,
            dropna=False,
            as_index=False,
        )[value_column]
        .sum()
        .sort_values(
            by=value_column,
            ascending=False,
        ) # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )

    return result


def group_and_average(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group by a column and calculate averages."""

    validate_column(
        df,
        group_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    if group_column == value_column:

        result = (
            df.groupby(
                group_column,
                dropna=False,
            )
            .agg(
                **{
                    "Average": (
                        value_column,
                        "mean",
                    )
                }
            )
            .reset_index()
        )

        return (
            result
            .sort_values(
                by="Average",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    result = (
        df.groupby(
            group_column,
            dropna=False,
            as_index=False,
        )[value_column]
        .mean()
        .sort_values(
            by=value_column,
            ascending=False,
        ) # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )

    return result


def group_and_count(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Group by a column and count rows."""

    validate_column(
        df,
        group_column,
    )

    result = (
        df.groupby(
            group_column,
            dropna=False,
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    return (
        result
        .sort_values(
            by="Count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# TOP N
# ============================================================

def top_n(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top N groups ranked by summed value."""

    validate_column(
        df,
        group_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    if (
        isinstance(n, bool)
        or not isinstance(
            n,
            numbers.Integral,
        )
    ):
        raise ValueError(
            "n must be an integer."
        )

    if n <= 0:
        raise ValueError(
            "n must be greater than zero."
        )

    result = group_and_sum(
        df,
        group_column,
        value_column,
    )

    return (
        result
        .head(int(n))
        .reset_index(drop=True)
    )


# ============================================================
# PERCENTAGE OF TOTAL
# ============================================================

def _calculate_percentage_and_sort(
    grouped_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Calculate percentages and sort descending."""

    total = grouped_df[value_col].sum()

    if total == 0:
        grouped_df["Percentage"] = 0.0
    else:
        grouped_df["Percentage"] = (
            grouped_df[value_col]
            / total
            * 100
        )

    return (
        grouped_df
        .sort_values(
            by="Percentage",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def percentage_of_total(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Calculate each group's percentage of the total."""

    validate_column(
        df,
        group_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    if group_column == value_column:

        grouped = (
            df.groupby(
                group_column,
                dropna=False,
            )[value_column]
            .sum()
            .rename("Value")
            .reset_index()
        )

        return _calculate_percentage_and_sort(
            grouped,
            "Value",
        )

    grouped = (
        df.groupby(
            group_column,
            dropna=False,
            as_index=False,
        )[value_column]
        .sum()
    )

    return _calculate_percentage_and_sort(
        grouped, # pyright: ignore[reportArgumentType]
        value_column,
    )


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Calculate monthly sums for a numeric value column."""

    validate_column(
        df,
        date_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    working_df = df.copy()

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[date_column]
    )

    if working_df.empty:
        raise ValueError(
            f"Column '{date_column}' contains no valid dates."
        )

    working_df["Month"] = (
        working_df[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        working_df
        .groupby(
            "Month",
            as_index=False,
        )[value_column]
        .sum()
    )

    return (
        result
        .sort_values("Month") # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )


# ============================================================
# VALUE COUNTS
# ============================================================

def value_counts(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return counts for each distinct value."""

    validate_column(
        df,
        column,
    )

    result = (
        df[column]
        .value_counts(
            dropna=False
        )
        .rename("Count")
        .reset_index()
    )

    result.columns = [
        column,
        "Count",
    ]

    return result


# ============================================================
# FILTERED AGGREGATIONS
# ============================================================

def filtered_sum(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> float:
    """Calculate a sum after applying filters."""

    validate_numeric_column(
        df,
        value_column,
    )

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return float(
        filtered_df[value_column].sum()
    )


def filtered_average(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> float:
    """Calculate an average after applying filters."""

    validate_numeric_column(
        df,
        value_column,
    )

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = (
        filtered_df[value_column]
        .dropna()
    )

    if values.empty:
        raise ValueError(
            f"No numeric values remain in "
            f"'{value_column}' after filtering."
        )

    return float(
        values.mean()
    )


def filtered_count(
    df: pd.DataFrame,
    filters: list,
    count_column: str | None = None,
) -> int:
    """Count filtered rows or non-null values."""

    filtered_df = apply_filters(
        df,
        filters,
    )

    if count_column is None:
        return len(filtered_df)

    validate_column(
        df,
        count_column,
    )

    return int(
        filtered_df[count_column].count()
    )


def filtered_unique_count(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> int:
    """Count unique values after applying filters."""

    validate_column(
        df,
        value_column,
    )

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        return 0

    return int(
        filtered_df[value_column]
        .nunique(dropna=True)
    )


def filtered_min(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> Any:
    """Return the minimum value after filtering."""

    validate_column(
        df,
        value_column,
    )

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = (
        filtered_df[value_column]
        .dropna()
    )

    if values.empty:
        raise ValueError(
            f"No values remain in "
            f"'{value_column}' after filtering."
        )

    return values.min()


def filtered_max(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> Any:
    """Return the maximum value after filtering."""

    validate_column(
        df,
        value_column,
    )

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = (
        filtered_df[value_column]
        .dropna()
    )

    if values.empty:
        raise ValueError(
            f"No values remain in "
            f"'{value_column}' after filtering."
        )

    return values.max()


# ============================================================
# FILTERED GROUP OPERATIONS
# ============================================================

def filtered_group_and_sum(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group and sum after applying filters."""

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return group_and_sum(
        filtered_df,
        group_column,
        value_column,
    )


def filtered_group_and_average(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group and average after applying filters."""

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return group_and_average(
        filtered_df,
        group_column,
        value_column,
    )


def filtered_value_counts(
    df: pd.DataFrame,
    filters: list,
    column: str,
) -> pd.DataFrame:
    """Return value counts after applying filters."""

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return value_counts(
        filtered_df,
        column,
    )


def filtered_top_n(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return top N groups after applying filters."""

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return top_n(
        filtered_df,
        group_column,
        value_column,
        n,
    )


# ============================================================
# DATAFRAME SERIALIZATION HELPER
# ============================================================

def dataframe_to_records(
    result: Any,
) -> Any:
    """
    Convert pandas/numpy results into JSON-safe Python objects.

    Handles:
        DataFrame
        Series
        Timestamp
        Timedelta
        numpy scalars
        numpy arrays
        dict
        list
        tuple
        NaN / NaT
    """

    if isinstance(
        result,
        pd.DataFrame,
    ):
        return [
            dataframe_to_records(row)
            for row in result.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        result,
        pd.Series,
    ):
        return {
            str(key): dataframe_to_records(
                value
            )
            for key, value in result.to_dict().items()
        }

    if isinstance(
        result,
        pd.Timestamp,
    ):
        return result.isoformat()

    if isinstance(
        result,
        pd.Timedelta,
    ):
        return result.total_seconds()

    if isinstance(
        result,
        np.generic,
    ):
        return dataframe_to_records(
            result.item()
        )

    if isinstance(
        result,
        np.ndarray,
    ):
        return [
            dataframe_to_records(item)
            for item in result.tolist()
        ]

    if isinstance(
        result,
        dict,
    ):
        return {
            str(key): dataframe_to_records(value)
            for key, value in result.items()
        }

    if isinstance(
        result,
        (list, tuple),
    ):
        return [
            dataframe_to_records(item)
            for item in result
        ]

    if result is None:
        return None

    # Handle pandas/numpy missing values.
    with contextlib.suppress(
        TypeError,
        ValueError,
    ):
        missing = pd.isna(result)

        if isinstance(
            missing,
            (bool, np.bool_),
        ) and missing:
            return None

    # Convert numpy-like numeric values and other
    # standard scalar values to ordinary Python objects.
    if isinstance(
        result,
        numbers.Number,
    ):
        if isinstance(
            result,
            complex,
        ):
            return result

        return result.item() if hasattr( # pyright: ignore[reportAttributeAccessIssue]
            result,
            "item",
        ) else result

    # Basic JSON-safe scalar.
    return result


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def normalize_filters(
    df: pd.DataFrame,
    filters: list,
) -> list:
    """
    Backward-compatible wrapper.

    The canonical normalize_filters implementation lives in
    analyst.py. This wrapper keeps older imports working without
    duplicating the normalization logic.
    """

    from analyst import normalize_filters as _normalize_filters

    return _normalize_filters(
        df,
        filters, # pyright: ignore[reportArgumentType]
    )

