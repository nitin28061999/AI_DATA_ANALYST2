import pandas as pd


# ============================================================
# SUPPORTED FILTER OPERATORS
# ============================================================

FILTER_OPERATORS = {
    "=",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
}


# ============================================================
# OPERATOR NORMALIZATION
# ============================================================

def normalize_operator(operator):
    """
    Normalize user/Gemini filter operators to the operators
    supported by the analysis engine.
    """

    if operator is None:
        return "="

    normalized = str(operator).strip().lower()

    # Gemini can occasionally repeat the equality character.
    # Any operator consisting only of "=" characters becomes "=".
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
    }

    return aliases.get(
        normalized,
        normalized,
    )


# ============================================================
# DATAFRAME VALIDATION
# ============================================================

def validate_dataframe(df):
    """
    Validate that the supplied object is a non-empty
    pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a valid pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Provided DataFrame is empty."
        )

    return True


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_column(df, column):
    """
    Validate that a column exists in the DataFrame.
    """

    validate_dataframe(df)

    if column is None or not str(column).strip():
        raise ValueError(
            "Column name cannot be empty."
        )

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    return True


# ============================================================
# NUMERIC COLUMN VALIDATION
# ============================================================

def validate_numeric_column(df, column):
    """
    Validate that a column exists and contains
    usable numeric data.
    """

    validate_column(
        df,
        column,
    )

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        raise ValueError(
            f"Column '{column}' does not contain "
            "usable numeric values."
        )

    return True


# ============================================================
# FILTER OPERATOR VALIDATION
# ============================================================

def validate_operator(operator):
    """
    Normalize and validate a filter operator.

    Returns the normalized operator.
    """

    normalized = normalize_operator(
        operator
    )

    if normalized not in FILTER_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator "
            f"'{normalized}'. "
            f"Supported operators: "
            f"{sorted(FILTER_OPERATORS)}"
        )

    return normalized


# ============================================================
# SINGLE FILTER VALIDATION
# ============================================================

def validate_filter(df, filter_item):
    """
    Validate one filter condition.

    Expected structure:

    {
        "column": "Revenue",
        "operator": ">=",
        "value": 100
    }

    Returns a normalized filter dictionary.
    """

    validate_dataframe(df)

    if not isinstance(filter_item, dict):
        raise ValueError(
            "Each filter must be an object."
        )

    column = filter_item.get("column")

    if not column:
        raise ValueError(
            "Filter is missing 'column'."
        )

    validate_column(
        df,
        column,
    )

    if "value" not in filter_item:
        raise ValueError(
            f"Filter for '{column}' "
            "is missing 'value'."
        )

    operator = validate_operator(
        filter_item.get(
            "operator",
            "=",
        )
    )

    return {
        "column": column,
        "operator": operator,
        "value": filter_item.get("value"),
    }


# ============================================================
# FILTER VALIDATION
# ============================================================

def validate_filters(df, filters):
    """
    Validate multiple filter conditions.

    Returns a normalized list of filters.
    """

    validate_dataframe(df)

    if filters is None:
        return []

    if not isinstance(filters, list):
        raise ValueError(
            "'filters' must be a list."
        )

    normalized_filters = []

    for filter_item in filters:
        normalized_filter = validate_filter(
            df,
            filter_item,
        )

        normalized_filters.append(
            normalized_filter
        )

    return normalized_filters


# ============================================================
# OPERATION VALIDATION
# ============================================================

SUPPORTED_OPERATIONS = {
    "calculate_sum",
    "calculate_average",
    "calculate_count",
    "calculate_unique_count",
    "calculate_min",
    "calculate_max",

    "group_and_sum",
    "group_and_average",
    "group_and_count",

    "top_n",
    "value_counts",

    "filtered_sum",
    "filtered_average",
    "filtered_count",
    "filtered_unique_count",
    "filtered_min",
    "filtered_max",

    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_value_counts",
    "filtered_top_n",
}


def validate_operation(operation):
    """
    Validate that an analysis operation is supported.
    """

    if not operation:
        raise ValueError(
            "Analysis plan is missing 'operation'."
        )

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: "
            f"'{operation}'. "
            f"Supported operations: "
            f"{sorted(SUPPORTED_OPERATIONS)}"
        )

    return True


# ============================================================
# PLAN VALIDATION
# ============================================================

def validate_plan(df, plan):
    """
    Validate the structural parameters of an analysis plan.

    This does not execute the plan.
    """

    validate_dataframe(df)

    if not isinstance(plan, dict):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = plan.get("operation")

    validate_operation(
        operation
    )

    # --------------------------------------------------------
    # Single-column operations
    # --------------------------------------------------------

    if operation in {
        "calculate_sum",
        "calculate_average",
        "calculate_unique_count",
        "calculate_min",
        "calculate_max",
        "value_counts",
    }:
        column = plan.get("column")

        if not column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'column'."
            )

        validate_column(
            df,
            column,
        )

    # --------------------------------------------------------
    # Numeric single-column operations
    # --------------------------------------------------------

    if operation in {
        "calculate_sum",
        "calculate_average",
        "calculate_min",
        "calculate_max",
    }:
        validate_numeric_column(
            df,
            plan.get("column"),
        )

    # --------------------------------------------------------
    # Count operation
    # --------------------------------------------------------

    if operation == "calculate_count":
        column = plan.get("column")

        if column is not None:
            validate_column(
                df,
                column,
            )

    # --------------------------------------------------------
    # Group operations
    # --------------------------------------------------------

    if operation in {
        "group_and_sum",
        "group_and_average",
        "group_and_count",
    }:
        group_column = plan.get(
            "group_column"
        )

        if not group_column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'group_column'."
            )

        validate_column(
            df,
            group_column,
        )

    if operation in {
        "group_and_sum",
        "group_and_average",
    }:
        value_column = plan.get(
            "value_column"
        )

        if not value_column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'value_column'."
            )

        validate_numeric_column(
            df,
            value_column,
        )

    # --------------------------------------------------------
    # Top N
    # --------------------------------------------------------

    if operation == "top_n":
        _extracted_from_validate_plan_122(
            plan,
            "Operation 'top_n' " "requires 'group_column'.",
            "Operation 'top_n' " "requires 'value_column'.",
            df,
        )
    # --------------------------------------------------------
    # Filtered operations
    # --------------------------------------------------------

    if operation.startswith("filtered_"): # pyright: ignore[reportOptionalMemberAccess]
        filters = plan.get(
            "filters",
            [],
        )

        normalized_filters = validate_filters(
            df,
            filters,
        )

        # Update the plan in-place with normalized filters.
        plan["filters"] = normalized_filters

    # --------------------------------------------------------
    # Filtered value column operations
    # --------------------------------------------------------

    if operation in {
        "filtered_sum",
        "filtered_average",
        "filtered_unique_count",
        "filtered_min",
        "filtered_max",
    }:
        value_column = plan.get(
            "value_column"
        )

        if not value_column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'value_column'."
            )

        validate_column(
            df,
            value_column,
        )

    if operation in {
        "filtered_sum",
        "filtered_average",
        "filtered_min",
        "filtered_max",
    }:
        validate_numeric_column(
            df,
            plan.get("value_column"),
        )

    # --------------------------------------------------------
    # Filtered count
    # --------------------------------------------------------

    if operation == "filtered_count":
        count_column = plan.get(
            "count_column"
        )

        if count_column is not None:
            validate_column(
                df,
                count_column,
            )

    # --------------------------------------------------------
    # Filtered group operations
    # --------------------------------------------------------

    if operation in {
        "filtered_group_and_sum",
        "filtered_group_and_average",
    }:
        group_column = plan.get(
            "group_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not group_column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'group_column'."
            )

        if not value_column:
            raise ValueError(
                f"Operation '{operation}' "
                "requires 'value_column'."
            )

        validate_column(
            df,
            group_column,
        )

        validate_numeric_column(
            df,
            value_column,
        )

    # --------------------------------------------------------
    # Filtered value counts
    # --------------------------------------------------------

    if operation == "filtered_value_counts":
        column = plan.get(
            "column"
        )

        if not column:
            raise ValueError(
                "Operation 'filtered_value_counts' "
                "requires 'column'."
            )

        validate_column(
            df,
            column,
        )

    # --------------------------------------------------------
    # Filtered Top N
    # --------------------------------------------------------

    if operation == "filtered_top_n":
        _extracted_from_validate_plan_122(
            plan,
            "Operation 'filtered_top_n' " "requires 'group_column'.",
            "Operation 'filtered_top_n' " "requires 'value_column'.",
            df,
        )
    return True


# TODO Rename this here and in `validate_plan`
def _extracted_from_validate_plan_122(plan, arg1, arg2, df):
    group_column = plan.get(
        "group_column"
    )

    value_column = plan.get(
        "value_column"
    )

    if not group_column:
        raise ValueError(arg1)

    if not value_column:
        raise ValueError(arg2)

    validate_column(
        df,
        group_column,
    )

    validate_numeric_column(
        df,
        value_column,
    )

    n = plan.get(
        "n",
        5,
    )

    if not isinstance(n, int):
        raise ValueError(
            "'n' must be an integer."
        )

    if n <= 0:
        raise ValueError(
            "'n' must be greater than zero."
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def validate(df, plan):
    """
    Backward-compatible validation entry point.
    """

    return validate_plan(
        df,
        plan,
    )