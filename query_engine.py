from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import analysis_tools as tools


# ============================================================
# QUERY-PLAN VALIDATION
# ============================================================

COLUMN_OPS = {
    "calculate_sum",
    "sum",
    "calculate_average",
    "average",
    "mean",
    "calculate_min",
    "min",
    "calculate_max",
    "max",
    "calculate_unique_count",
    "unique_count",
    "value_counts",
    "filtered_value_counts",
}

GROUP_OPS = {
    "group_and_sum",
    "group_sum",
    "filtered_group_sum",
    "group_and_average",
    "group_average",
    "filtered_group_average",
    "group_and_count",
    "group_count",
    "top_n",
    "filtered_top_n",
    "percentage_of_total",
    "filtered_percentage_of_total",
}

GROUP_VALUE_OPS = {
    "group_and_sum",
    "group_sum",
    "filtered_group_sum",
    "group_and_average",
    "group_average",
    "filtered_group_average",
    "top_n",
    "filtered_top_n",
    "percentage_of_total",
    "filtered_percentage_of_total",
}

MONTHLY_VALUE_OPS = {
    "monthly_sum",
    "filtered_monthly_sum",
    "monthly_average",
    "filtered_monthly_average",
}

MONTHLY_OPS = MONTHLY_VALUE_OPS | {
    "monthly_count",
    "filtered_monthly_count",
}

VALID_OPERATIONS = COLUMN_OPS | GROUP_OPS | MONTHLY_OPS | {
    "calculate_count",
    "count",
}


def _first_present(plan: Dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty value from a query plan."""
    for key in keys:
        value = plan.get(key)
        if value is not None and value != "":
            return value
    return None


def validate_query(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an analysis query plan."""
    if not isinstance(plan, dict):
        raise ValueError("Query must be a dictionary.")

    operation = plan.get("operation")
    if not operation:
        raise ValueError("Analysis plan missing 'operation' field.")

    op = str(operation).strip().lower()

    if op not in VALID_OPERATIONS:
        raise ValueError(f"Unsupported query operation: '{operation}'")

    column = _first_present(
        plan,
        "column",
        "value_column",
        "count_column",
    )
    group_column = _first_present(
        plan,
        "group_column",
        "group_by",
    )

    if op in COLUMN_OPS and not column:
        raise ValueError(
            f"Operation '{operation}' missing 'column'."
        )

    if op in GROUP_OPS and not group_column:
        raise ValueError(
            f"Operation '{operation}' missing 'group_by'."
        )

    if op in GROUP_VALUE_OPS and not column:
        raise ValueError(
            f"Operation '{operation}' requires 'value_column'."
        )

    if op in MONTHLY_OPS:
        date_column = _first_present(
            plan,
            "date_column",
            "date",
        ) or "Date"

        if not date_column:
            raise ValueError(
                f"Operation '{operation}' requires 'date_column'."
            )

        if op in MONTHLY_VALUE_OPS and not column:
            raise ValueError(
                f"Operation '{operation}' requires 'value_column'."
            )

    if op in {"top_n", "filtered_top_n"}:
        if "n" not in plan:
            raise ValueError(
                "Operation 'top_n' missing 'n'."
            )

    # If filters are supplied, keep their validation delegated to the
    # analysis-tools filter engine, which owns the canonical filter rules.
    filters = plan.get("filters", [])
    if filters is not None and not isinstance(filters, list):
        raise ValueError("Filters must be a list.")

    plan["operation"] = op
    return plan


# ============================================================
# MONTHLY HELPERS
# ============================================================


def _prepare_monthly_dataframe(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    """Validate, optionally filter, and prepare data for monthly analysis."""
    tools.validate_column(df, date_column)

    if value_column is not None:
        tools.validate_numeric_column(df, value_column)

    working_df = (
        tools.apply_filters(df, filters)
        if filters
        else df.copy()
    )

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )

    working_df = working_df.dropna(subset=[date_column])

    if working_df.empty:
        raise ValueError(
            f"Column '{date_column}' contains no valid dates."
        )

    working_df["Month"] = (
        working_df[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    return working_df


def _monthly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    filters: list | None = None,
) -> pd.DataFrame:
    """Calculate monthly sums, with optional filters."""
    working_df = _prepare_monthly_dataframe(
        df,
        date_column,
        value_column,
        filters,
    )

    result = (
        working_df
        .groupby("Month", as_index=False)[value_column]
        .sum()
    )

    return (
        result
        .sort_values("Month") # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )


def _monthly_average(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    filters: list | None = None,
) -> pd.DataFrame:
    """Calculate monthly averages, with optional filters."""
    working_df = _prepare_monthly_dataframe(
        df,
        date_column,
        value_column,
        filters,
    )

    result = (
        working_df
        .groupby("Month", as_index=False)[value_column]
        .mean()
    )

    return (
        result
        .sort_values("Month") # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )


def _monthly_count(
    df: pd.DataFrame,
    date_column: str,
    filters: list | None = None,
) -> pd.DataFrame:
    """Count rows per month, with optional filters."""
    working_df = _prepare_monthly_dataframe(
        df,
        date_column,
        filters=filters,
    )

    result = (
        working_df
        .groupby("Month")
        .size()
        .reset_index(name="Count")
    )

    return (
        result
        .sort_values("Month")
        .reset_index(drop=True)
    )


def _filtered_group_count(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
) -> pd.DataFrame:
    """Group-count rows after applying filters."""
    filtered_df = tools.apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return tools.group_and_count(
        filtered_df,
        group_column,
    )


def _filtered_percentage_of_total(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Calculate group percentages after applying filters."""
    filtered_df = tools.apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return tools.percentage_of_total(
        filtered_df,
        group_column,
        value_column,
    )


# ============================================================
# QUERY EXECUTION
# ============================================================


def execute_query(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:  # sourcery skip: low-code-quality
    """Execute a validated analysis plan against a pandas DataFrame."""
    validated_plan = validate_query(plan)
    operation = validated_plan["operation"]

    column = _first_present(
        validated_plan,
        "column",
        "value_column",
        "count_column",
    )
    group_column = _first_present(
        validated_plan,
        "group_column",
        "group_by",
    )
    date_column = _first_present(
        validated_plan,
        "date_column",
        "date",
    ) or "Date"
    filters = validated_plan.get("filters", [])
    n = validated_plan.get("n", 5)
    has_filters = bool(filters)

    match operation:
        # ----------------------------------------------------
        # BASIC AGGREGATIONS
        # ----------------------------------------------------
        case "calculate_sum" | "sum":
            return (
                tools.filtered_sum(df, filters, column)
                if has_filters
                else tools.calculate_sum(df, column)
            )

        case "calculate_average" | "average" | "mean":
            return (
                tools.filtered_average(df, filters, column)
                if has_filters
                else tools.calculate_average(df, column)
            )

        case "calculate_count" | "count":
            return (
                tools.filtered_count(df, filters, column)
                if has_filters
                else tools.calculate_count(df, column)
            )

        case "calculate_unique_count" | "unique_count":
            return (
                tools.filtered_unique_count(df, filters, column)
                if has_filters
                else tools.calculate_unique_count(df, column)
            )

        case "calculate_min" | "min":
            return (
                tools.filtered_min(df, filters, column)
                if has_filters
                else tools.calculate_min(df, column)
            )

        case "calculate_max" | "max":
            return (
                tools.filtered_max(df, filters, column)
                if has_filters
                else tools.calculate_max(df, column)
            )

        # ----------------------------------------------------
        # GROUP OPERATIONS
        # ----------------------------------------------------
        case "group_and_sum" | "group_sum" | "filtered_group_sum":
            return (
                tools.filtered_group_and_sum(
                    df,
                    filters,
                    group_column,
                    column,
                )
                if has_filters
                else tools.group_and_sum(
                    df,
                    group_column,
                    column,
                )
            )

        case "group_and_average" | "group_average" | "filtered_group_average":
            return (
                tools.filtered_group_and_average(
                    df,
                    filters,
                    group_column,
                    column,
                )
                if has_filters
                else tools.group_and_average(
                    df,
                    group_column,
                    column,
                )
            )

        case "group_and_count" | "group_count":
            return (
                _filtered_group_count(
                    df,
                    filters,
                    group_column,
                )
                if has_filters
                else tools.group_and_count(
                    df,
                    group_column,
                )
            )

        case "value_counts" | "filtered_value_counts":
            return (
                tools.filtered_value_counts(df, filters, column)
                if has_filters
                else tools.value_counts(df, column)
            )

        case "top_n" | "filtered_top_n":
            return (
                tools.filtered_top_n(
                    df,
                    filters,
                    group_column,
                    column,
                    n,
                )
                if has_filters
                else tools.top_n(
                    df,
                    group_column,
                    column,
                    n,
                )
            )

        case "percentage_of_total" | "filtered_percentage_of_total":
            return (
                _filtered_percentage_of_total(
                    df,
                    filters,
                    group_column,
                    column,
                )
                if has_filters
                else tools.percentage_of_total(
                    df,
                    group_column,
                    column,
                )
            )

        # ----------------------------------------------------
        # MONTHLY OPERATIONS
        # ----------------------------------------------------
        case "monthly_sum" | "filtered_monthly_sum":
            return _monthly_sum(
                df,
                date_column,
                column,
                filters if has_filters else None,
            )

        case "monthly_average" | "filtered_monthly_average":
            return _monthly_average(
                df,
                date_column,
                column,
                filters if has_filters else None,
            )

        case "monthly_count" | "filtered_monthly_count":
            return _monthly_count(
                df,
                date_column,
                filters if has_filters else None,
            )

        case _:
            # validate_query() makes this unreachable, but retaining the
            # explicit guard keeps execute_query safe if the operation set
            # changes in the future.
            raise ValueError(
                f"Unsupported query operation: '{operation}'"
            )