# analyst.py

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

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
    percentage_of_total,
    monthly_sum,
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
)
class GeminiFilter(BaseModel):
    column: str = Field(
        description="Actual dataset column name."
    )

    operator: str = Field(
        description=(
            "Filter operator. Must be one of: "
            "=, !=, >, >=, <, <=, contains, between."
        )
    )

    value: Union[
        str,
        float,
        int,
        bool,
        List[Any],
    ] = Field(
        description=(
            "Filter value. For between, provide exactly "
            "two values in a list."
        )
    )


class GeminiPlan(BaseModel):
    operation: str = Field(
        description=(
            "Analysis operation. Must be one of the "
            "supported operations supplied in the prompt."
        )
    )

    column: Optional[str] = Field(
        default=None,
        description="Dataset column used by a simple column operation.",
    )

    group_column: Optional[str] = Field(
        default=None,
        description="Dataset column used for grouping.",
    )

    value_column: Optional[str] = Field(
        default=None,
        description="Dataset numeric/value column.",
    )

    count_column: Optional[str] = Field(
        default=None,
        description="Dataset column used for counting.",
    )

    date_column: Optional[str] = Field(
        default=None,
        description="Dataset date/datetime column.",
    )

    n: Optional[int] = Field(
        default=None,
        description="Number of rows/groups requested for top-N operations.",
    )

    filters: Optional[List[GeminiFilter]] = Field(
        default=None,
        description="Optional list of dataset filters.",
    )

# ============================================================
# GEMINI CONFIGURATION
# ============================================================
#
# Gemini is REQUIRED for this analyst.
#
# The architecture is:
#
#     User question
#           |
#           v
#     Gemini chooses the analysis plan
#           |
#           v
#     Python validates the plan
#           |
#           v
#     Python performs the calculation
#           |
#           v
#     Gemini explains the actual result
#
# No ANALYST_USE_GEMINI switch is used.
# A missing SDK or API key is treated as a configuration error.
#
# ============================================================

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()

if not MODEL_NAME:
    raise ValueError(
        "GEMINI_MODEL cannot be empty."
    )

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

client = None

def get_gemini_client():
    """Create the Gemini client only when an AI request is actually made."""
    global client
    if client is not None:
        return client
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK is not installed. Install the package from requirements.txt."
        ) from exc
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing from the environment or .env file."
        )
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


# ============================================================
# SUPPORTED OPERATIONS
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
    "percentage_of_total",
    "monthly_sum",
    "value_counts",
    "filtered_sum",
    "filtered_average",
    "filtered_count",
    "filtered_unique_count",
    "filtered_min",
    "filtered_max",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_group_and_count",
    "filtered_value_counts",
    "filtered_top_n",
}


COLUMN_OPS = {
    "calculate_sum",
    "calculate_average",
    "calculate_count",
    "calculate_unique_count",
    "calculate_min",
    "calculate_max",
    "value_counts",
}


GROUP_VALUE_OPS = {
    "group_and_sum",
    "group_and_average",
    "top_n",
    "percentage_of_total",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_top_n",
}


FILTERED_OPS = {
    "filtered_sum",
    "filtered_average",
    "filtered_count",
    "filtered_unique_count",
    "filtered_min",
    "filtered_max",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_group_and_count",
    "filtered_value_counts",
    "filtered_top_n",
}


FILTER_OPERATORS = {
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


# Gemini structured-output schema. The schema constrains the model to
# return a single analysis plan while Python still performs the final
# column/operator validation against the real DataFrame.
GEMINI_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": sorted(SUPPORTED_OPERATIONS),
        },
        "column": {
            "type": ["string", "null"],
        },
        "group_column": {
            "type": ["string", "null"],
        },
        "value_column": {
            "type": ["string", "null"],
        },
        "count_column": {
            "type": ["string", "null"],
        },
        "date_column": {
            "type": ["string", "null"],
        },
        "n": {
            "type": ["integer", "null"],
        },
        "filters": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {
                        "type": "string",
                        "enum": sorted(FILTER_OPERATORS),
                    },
                    "value": {
                        "type": [
                            "string",
                            "number",
                            "boolean",
                            "array",
                            "null",
                        ],
                    },
                },
                "required": [
                    "column",
                    "operator",
                    "value",
                ],
            },
        },
    },
    "required": ["operation"],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    """Validate that df is a usable pandas DataFrame."""

    if df is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The dataset has no columns."
        )


def get_dataset_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return dataset column names as strings."""

    validate_dataframe(df)

    return [
        str(column)
        for column in df.columns
    ]


def _norm(
    value: Any,
) -> str:
    """Normalize text for matching."""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .strip()
        .lower(),
    ).strip()


def _safe(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values into
    JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _safe(item)
            for item in value
        ]

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return [
            _safe(record)
            for record in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        value,
        pd.Series,
    ):
        return {
            str(key): _safe(item)
            for key, item in value.to_dict().items()
        }

    if hasattr(
        value,
        "item",
    ):
        with contextlib.suppress(
            Exception
        ):
            return _safe(
                value.item()
            )

    if hasattr(
        value,
        "isoformat",
    ):
        with contextlib.suppress(
            Exception
        ):
            return value.isoformat()

    try:
        json.dumps(value)
        return value

    except Exception:
        return str(value)


def serialize_result(
    result: Any,
) -> Any:
    """Public result serialization helper."""

    return _safe(result)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(
    text: str,
) -> Dict[str, Any]:
    """Extract a JSON object from model output."""

    if not text:
        raise ValueError(
            "Empty JSON response."
        )

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```",
        "",
        text,
    )

    text = text.strip()

    with contextlib.suppress(
        json.JSONDecodeError
    ):
        value = json.loads(text)

        if isinstance(
            value,
            dict,
        ):
            return value

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            "No JSON object found in model response."
        )

    json_text = text[
        start : end + 1
    ]

    try:
        value = json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse model JSON: "
            f"{exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a JSON object."
        )

    return value


# ============================================================
# PROFILE
# ============================================================

def build_profile(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build a profile from the real DataFrame."""

    validate_dataframe(df)

    columns = get_dataset_columns(df)

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    datetime_columns = [
        column
        for column in columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]

    text_columns = [
        column
        for column in columns
        if column not in numeric_columns
        and column not in datetime_columns
    ]

    return {
        "columns": columns,
        "row_count": len(df),
        "numeric_columns": numeric_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
    }


def normalize_profile(
    df: pd.DataFrame,
    profile: Optional[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Normalize a supplied profile.

    The real DataFrame is authoritative.
    """

    validate_dataframe(df)

    generated = build_profile(df)

    if profile is None:
        return generated

    if not isinstance(
        profile,
        dict,
    ):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    result = dict(profile)

    # Keep the rich column metadata supplied by data_profile.py, while
    # exposing an authoritative list of real DataFrame column names.
    # The actual DataFrame always wins for column validation.
    result["column_names"] = generated[
        "columns"
    ]

    if "columns" not in result:
        result["columns"] = generated[
            "columns"
        ]

    for key, value in generated.items():
        result.setdefault(
            key,
            value,
        )

    return result


def _profile_columns(
    profile: Optional[
        Dict[str, Any]
    ],
) -> List[str]:
    """Return normalized profile columns."""

    if profile is None:
        return []

    if not isinstance(
        profile,
        dict,
    ):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    columns = profile.get(
        "column_names",
        profile.get("columns", []),
    )

    if isinstance(
        columns,
        dict,
    ):
        columns = list(
            columns.keys()
        )

    if isinstance(
        columns,
        (list, tuple),
    ):
        normalized = []

        for column in columns:
            if isinstance(column, dict):
                name = column.get("name")
                if name is not None:
                    normalized.append(str(name))
            else:
                normalized.append(str(column))

        return normalized

    return []


# ============================================================
# COLUMN SELECTION
# ============================================================

def _find_column(
    question: str,
    columns: List[str],
) -> Optional[str]:
    """
    Find an explicitly mentioned column.

    Longest names are checked first.
    """

    q = _norm(question)

    return next(
        (
            column
            for column in sorted(
                columns,
                key=len,
                reverse=True,
            )
            if _norm(column) in q
        ),
        None,
    )


def _semantic_column(
    columns: List[str],
    keywords: List[str],
) -> Optional[str]:
    """Find the best semantic column match."""

    scored = []

    normalized_keywords = [
        _norm(keyword)
        for keyword in keywords
    ]

    for column in columns:
        name = _norm(column)
        score = 0

        for keyword in normalized_keywords:

            if name == keyword:
                score += 100

            elif keyword in name:
                score += 20

        if score:
            scored.append(
                (
                    score,
                    -len(column),
                    column,
                )
            )

    return None if not scored else max(scored)[2]


def _numeric_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return numeric columns."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


def _text_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return non-numeric columns."""

    numeric = set(
        _numeric_columns(df)
    )

    return [
        str(column)
        for column in df.columns
        if str(column) not in numeric
    ]


def _value_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose the numeric value column."""

    columns = get_dataset_columns(df)
    numeric = _numeric_columns(df)

    explicit = _find_column(
        question,
        columns,
    )

    if explicit in numeric:
        return explicit

    found = _semantic_column(
        numeric,
        [
            "revenue",
            "sales",
            "sale",
            "amount",
            "price",
            "profit",
            "income",
            "expense",
            "expenses",
            "salary",
            "score",
            "cost",
            "quantity",
            "units",
            "value",
            "total",
        ],
    )

    if found:
        return found

    return numeric[0] if numeric else None


def _group_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose the grouping column."""

    columns = get_dataset_columns(df)

    explicit = _find_column(
        question,
        columns,
    )

    if explicit:
        return explicit

    text = _text_columns(df)

    found = _semantic_column(
        text,
        [
            "city",
            "state",
            "country",
            "region",
            "product",
            "category",
            "department",
            "employee",
            "customer",
            "status",
            "brand",
            "segment",
            "type",
            "location",
        ],
    )

    if found:
        return found

    return text[0] if text else None


def _count_column(
    question: str,
    df: pd.DataFrame,
) -> str:
    """Choose the column whose rows should be counted."""

    columns = get_dataset_columns(df)
    q = _norm(question)

    candidates = [
        (
            "invoice",
            [
                "invoice",
                "invoice_id",
                "invoice id",
            ],
        ),
        (
            "order",
            [
                "order",
                "order_id",
                "order id",
            ],
        ),
        (
            "transaction",
            [
                "transaction",
                "transaction_id",
                "transaction id",
            ],
        ),
        (
            "customer",
            [
                "customer",
                "customer_id",
                "customer id",
            ],
        ),
        (
            "employee",
            [
                "employee",
                "employee_id",
                "employee id",
            ],
        ),
    ]

    for trigger, keys in candidates:

        if trigger not in q:
            continue

        found = _semantic_column(
            columns,
            keys,
        )

        if found:
            return found

    explicit = _find_column(
        question,
        columns,
    )

    return explicit or columns[0]


def _unique_column(
    question: str,
    df: pd.DataFrame,
) -> str:
    """Choose the column whose distinct values are counted."""

    columns = get_dataset_columns(df)
    q = _norm(question)

    candidates = [
        (
            "customer",
            [
                "customer",
                "customer_id",
                "customer id",
            ],
        ),
        (
            "product",
            [
                "product",
                "product_id",
                "product id",
            ],
        ),
        (
            "employee",
            [
                "employee",
                "employee_id",
                "employee id",
            ],
        ),
        (
            "invoice",
            [
                "invoice",
                "invoice_id",
                "invoice id",
            ],
        ),
        (
            "order",
            [
                "order",
                "order_id",
                "order id",
            ],
        ),
        (
            "transaction",
            [
                "transaction",
                "transaction_id",
                "transaction id",
            ],
        ),
        (
            "city",
            [
                "city",
            ],
        ),
    ]

    for trigger, keys in candidates:

        if trigger not in q:
            continue

        found = _semantic_column(
            columns,
            keys,
        )

        if found:
            return found

    explicit = _find_column(
        question,
        columns,
    )

    return explicit or columns[0]


# ============================================================
# FILTER HELPERS
# ============================================================

def _coerce(
    value: Any,
    series: pd.Series,
) -> Any:
    """Coerce text into the column's data type."""

    if not isinstance(
        value,
        str,
    ):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    if pd.api.types.is_numeric_dtype(
        series
    ):
        with contextlib.suppress(
            ValueError
        ):
            return float(value)

        with contextlib.suppress(
            ValueError
        ):
            return int(value)

    if pd.api.types.is_bool_dtype(
        series
    ):
        lowered = value.lower()

        if lowered in {
            "true",
            "yes",
        }:
            return True

        if lowered in {
            "false",
            "no",
        }:
            return False

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        with contextlib.suppress(
            Exception
        ):
            return pd.to_datetime(
                value
            )

    return value


def _dataset_value(
    column: str,
    value: Any,
    df: pd.DataFrame,
) -> Any:
    """
    Resolve a textual value against an
    actual dataset value.

    Example:
        delhi -> Delhi
        laptop -> Laptop
    """

    if not isinstance(
        value,
        str,
    ):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    series = df[column]

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mask = (
        normalized
        == value.lower()
    )

    return (
        series[mask].iloc[0]
        if mask.any()
        else _coerce(
            value,
            series,
        )
    )

def _find_column_for_value(
    value: str,
    df: pd.DataFrame,
    preferred_keywords: Optional[
        List[str]
    ] = None,
) -> Optional[str]:
    """
    Find a dataset column containing
    an exact textual value.
    """

    value_norm = _norm(value)

    columns = get_dataset_columns(df)

    preferred = (
        preferred_keywords or []
    )

    ordered_columns = sorted(
        columns,
        key=lambda column: (
            0
            if any(
                _norm(keyword)
                in _norm(column)
                for keyword in preferred
            )
            else 1,
            len(column),
        ),
    )

    for column in ordered_columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if value_norm in set(values):
            return column

    return None


# ============================================================
# FILTER EXTRACTION
# ============================================================

def _extract_filters(
    question: str,
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Extract deterministic filters.

    Supported examples:

        City = Delhi
        City is Delhi
        City == Delhi
        City != Delhi
        Price > 500
        Price >= 500
        Price < 500
        Price <= 500
        Product contains Laptop
        Salary between 50000 and 70000
        in Delhi
        for Laptop
        from Delhi
        City Delhi
        Product Laptop
    """

    columns = get_dataset_columns(df)
    q = question.strip()

    filters: List[
        Dict[str, Any]
    ] = []

    used_columns: set[str] = set()

    def add(
        column: str,
        operator: str,
        value: Any,
    ) -> None:

        if column in used_columns:
            return

        if isinstance(
            value,
            str,
        ):
            value = _dataset_value(
                column,
                value,
                df,
            )

        if operator == "==":
            operator = "="

        filters.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

        used_columns.add(column)

    # --------------------------------------------------------
    # Explicit column + BETWEEN
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s+between\s+"
            rf"(.+?)"
            rf"\s+and\s+"
            rf"(.+?)"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        first_value = match[1].strip()

        second_value = match[2].strip()

        add(
            column,
            "between",
            [
                _dataset_value(
                    column,
                    first_value,
                    df,
                ),
                _dataset_value(
                    column,
                    second_value,
                    df,
                ),
            ],
        )

    # --------------------------------------------------------
    # Explicit column + contains/is/operators
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s*"
            rf"(>=|<=|!=|==|=|>|<|contains|is)"
            rf"\s*"
            rf"(.+?)"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        operator = match[1].lower().strip()

        raw_value = match[2].strip()

        if operator == "is":
            operator = "="

        if operator == "contains":
            raw_value = (
                raw_value
                .strip("\"'")
                .strip()
            )

            add(
                column,
                "contains",
                raw_value,
            )

        else:
            add(
                column,
                operator,
                raw_value,
            )

    # --------------------------------------------------------
    # Explicit numeric comparisons
    #
    # Handles:
    # Price > 500
    # Salary >= 50000
    # Revenue < 1000
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s*"
            rf"(>=|<=|!=|>|<)"
            rf"\s*"
            rf"(-?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        add(column, match[1], _coerce(match[2], df[column]))

    # --------------------------------------------------------
    # Generic numeric BETWEEN
    #
    # Example:
    # revenue between 100 and 500
    # --------------------------------------------------------

    generic_between = re.search(
        r"\bbetween\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+and\s+"
        r"(-?\d+(?:\.\d+)?)",
        q,
        re.IGNORECASE,
    )

    if generic_between:

        value_column = _value_column(
            q,
            df,
        )

        if (
            value_column
            and value_column
            not in used_columns
        ):
            add(
                value_column,
                "between",
                [
                    _coerce(generic_between[1], df[value_column]),
                    _coerce(generic_between[2], df[value_column]),
                ],
            )

    # --------------------------------------------------------
    # Natural equality:
    #
    # in Delhi
    # for Laptop
    # from Delhi
    # --------------------------------------------------------

    natural_pattern = re.compile(
        r"\b(in|for|from)\s+"
        r"['\"]?"
        r"([^,;?.]+?)"
        r"['\"]?"
        r"(?=\s+(?:for|in|from|and|where)\b|[,;?.]|$)",
        re.IGNORECASE,
    )

    natural_matches = (
        natural_pattern.findall(q)
    )

    for keyword, raw_value in natural_matches:

        value = (
            raw_value
            .strip()
            .strip("\"'")
            .strip()
        )

        if not value:
            continue

        if keyword.lower() in {
            "in",
            "from",
        }:
            preferred_keywords = [
                "city",
                "state",
                "country",
                "region",
                "location",
            ]

        else:
            preferred_keywords = [
                "product",
                "category",
                "brand",
                "type",
                "segment",
                "department",
            ]

        column = _find_column_for_value(
            value,
            df,
            preferred_keywords,
        )

        if column:
            add(
                column,
                "=",
                value,
            )

    # --------------------------------------------------------
    # Natural equality:
    #
    # City Delhi
    # Product Laptop
    # Category Electronics
    # --------------------------------------------------------

    operation_words = {
        "revenue",
        "sales",
        "amount",
        "profit",
        "average",
        "avg",
        "mean",
        "total",
        "sum",
        "count",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "largest",
        "smallest",
    }

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s+"
            rf"['\"]?"
            rf"([^,;?.]+?)"
            rf"['\"]?"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?.]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        candidate = match[1].strip().strip("\"'").strip()

        if not candidate:
            continue

        if _norm(candidate) in operation_words:
            continue

        dataset_values = set(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if _norm(candidate) in dataset_values:
            add(
                column,
                "=",
                candidate,
            )

    return filters


# ============================================================
# FILTER NORMALIZATION
# ============================================================

def normalize_filters(
    filters: Any,
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Validate and normalize filters.

    Supports both historical call styles ``normalize_filters(filters, df)``
    and ``normalize_filters(df, filters)``. Unknown columns are skipped when
    a DataFrame is supplied, matching the analysis-tool contract.
    """
    if isinstance(filters, pd.DataFrame):
        filters, df = df, filters

    if not isinstance(filters, list):
        raise ValueError("Filters must be a list.")
    if not filters:
        raise ValueError("At least one filter is required.")

    aliases = {
        "==": "=", "===": "=", "eq": "=", "equals": "=", "equal": "=",
        "ne": "!=", "not equal": "!=", "not_equal": "!=",
        "gt": ">", "greater than": ">", "greater_than": ">",
        "gte": ">=", "greater than or equal": ">=", "greater_than_or_equal": ">=",
        "lt": "<", "less than": "<", "less_than": "<",
        "lte": "<=", "less than or equal": "<=", "less_than_or_equal": "<=",
    }
    supported = {"=", "!=", ">", ">=", "<", "<=", "contains", "between"}

    result: List[Dict[str, Any]] = []
    for index, item in enumerate(filters):
        if not isinstance(item, dict):
            raise ValueError(f"Filter #{index + 1} must be a dictionary.")
        if "column" not in item:
            raise ValueError(f"Filter #{index + 1} is missing 'column'.")
        if "value" not in item:
            raise ValueError(f"Filter #{index + 1} is missing 'value'.")

        column = item["column"]
        if not column:
            raise ValueError(f"Filter #{index + 1} has an empty column.")
        if df is not None and column not in df.columns:
            continue

        raw_op = str(item.get("operator", "=") or "=").strip().lower()
        operator = "=" if raw_op and set(raw_op) == {"="} else aliases.get(raw_op, raw_op)
        if operator not in supported:
            raise ValueError(f"Unsupported filter operator '{operator}'.")

        value = item["value"]
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("The 'between' filter requires exactly two values.")
            if df is not None:
                value = [_coerce(value[0], df[column]), _coerce(value[1], df[column])]
            else:
                value = list(value)
        elif operator == "contains":
            value = str(value).strip()
        elif df is not None:
            value = _coerce(value, df[column])

        result.append({"column": column, "operator": operator, "value": value})

    return result

# ============================================================
# PLAN VALIDATION
# ============================================================

def validate_plan_columns(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> None:
    """
    Validate every column referenced by
    the analysis plan.
    """

    validate_dataframe(df)

    columns = set(
        get_dataset_columns(df)
    )

    operation = plan.get(
        "operation"
    )

    def require(
        key: str,
    ) -> None:

        value = plan.get(key)

        if not value:
            raise ValueError(
                f"{operation} requires "
                f"'{key}'."
            )

        if value not in columns:
            raise ValueError(
                f"Column '{value}' does not exist. "
                f"Available columns: "
                f"{sorted(columns)}"
            )

    if operation in COLUMN_OPS:
        require("column")

    if operation in GROUP_VALUE_OPS:
        require("group_column")
        require("value_column")

    if operation == "group_and_count":
        require("group_column")

    if operation == "monthly_sum":
        require("date_column")
        require("value_column")

    if operation in FILTERED_OPS:

        filters = plan.get(
            "filters"
        )

        if (
            not isinstance(
                filters,
                list,
            )
            or not filters
        ):
            raise ValueError(
                f"{operation} requires "
                "'filters'."
            )

        for index, filter_item in enumerate(
            filters
        ):

            if not isinstance(
                filter_item,
                dict,
            ):
                raise ValueError(
                    f"Filter #{index + 1} "
                    "must be a dictionary."
                )

            filter_column = (
                filter_item.get(
                    "column"
                )
            )

            if not filter_column:
                raise ValueError(
                    f"Filter #{index + 1} "
                    "is missing 'column'."
                )

            if (
                filter_column
                not in columns
            ):
                raise ValueError(
                    f"Filter column "
                    f"'{filter_column}' "
                    "does not exist."
                )

            if "value" not in filter_item:
                raise ValueError(
                    f"Filter #{index + 1} "
                    "is missing 'value'."
                )

        if operation in {
            "filtered_sum",
            "filtered_average",
            "filtered_unique_count",
            "filtered_min",
            "filtered_max",
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        }:
            require("value_column")

        if operation == "filtered_count":
            require("count_column")

        if operation in {
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_group_and_count",
            "filtered_top_n",
        }:
            require("group_column")

        if operation == "filtered_value_counts":
            require("column")


def validate_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate an analysis plan against a profile."""

    if not isinstance(
        plan,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get(
            "operation",
            "",
        )
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)

    result[
        "operation"
    ] = operation

    if operation.startswith(
        "filtered_"
    ):
        result[
            "filters"
        ] = normalize_filters(
            result.get(
                "filters"
            )
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:

        try:
            n = int(
                result.get(
                    "n",
                    5,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result["n"] = n

    columns = _profile_columns(
        profile
    )

    if columns:

        for key in (
            "column",
            "group_column",
            "value_column",
            "count_column",
            "date_column",
        ):

            value = result.get(
                key
            )

            if (
                value
                and value not in columns
            ):
                raise ValueError(
                    f"Column '{value}' "
                    "does not exist in the dataset."
                )

        for filter_item in result.get(
            "filters",
            
        ) or[]:

            if (
                filter_item["column"]
                not in columns
            ):
                raise ValueError(
                    f"Filter column "
                    f"'{filter_item['column']}' "
                    "does not exist."
                )

    return result


def normalize_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize a plan
    against the real DataFrame.
    """

    validate_dataframe(df)

    if not isinstance(
        plan,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get(
            "operation",
            "",
        )
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)

    result[
        "operation"
    ] = operation

    if operation.startswith(
        "filtered_"
    ):
        result[
            "filters"
        ] = normalize_filters(
            result.get(
                "filters"
            ),
            df=df,
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:

        try:
            n = int(
                result.get(
                    "n",
                    5,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result["n"] = n

    validate_plan_columns(
        df,
        result,
    )

    return result


# ============================================================
# TOP N EXTRACTION
# ============================================================

def _extract_n(
    question: str,
    default: int = 5,
) -> int:
    """Extract N from top/first/best/highest/bottom N."""

    match = re.search(
        r"\b(?:top|first|best|highest|bottom)\s+(\d+)\b",
        question,
        re.IGNORECASE,
    )

    return default if not match else max(1, int(match[1]))


def _has_grouping_intent(
    question: str,
) -> bool:
    """Determine whether the question requests grouped output."""

    q = _norm(question)

    return any(
        phrase in q
        for phrase in (
            " by ",
            "group by",
            "grouped by",
            "per city",
            "per product",
            "per category",
            "per department",
            "per region",
            "per state",
            "per country",
        )
    )


# ============================================================
# DETERMINISTIC PLANNER
# ============================================================

def deterministic_plan(
    question: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:  # sourcery skip: low-code-quality
    """
    Create an analysis plan without
    using an external API.
    """

    validate_dataframe(df)

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    q = _norm(question)

    filters = _extract_filters(
        question,
        df,
    )

    filtered = bool(filters)

    grouping_intent = (
        _has_grouping_intent(
            question
        )
    )

    # --------------------------------------------------------
    # TOP N / HIGHEST / BEST
    # --------------------------------------------------------

    explicit_top = (
        re.search(
            r"\btop\s+\d+\b",
            q,
        )
        is not None
    )

    top_words = (
        "highest",
        "largest",
        "greatest",
        "best performing",
        "top ",
        "maximum by",
    )

    top_intent = (
        explicit_top
        or any(
            phrase in q
            for phrase in top_words
        )
    )

    # "highest revenue" by itself is a maximum.
    # "highest revenue by city" is top-N grouped.
    if top_intent and (
        grouping_intent
        or explicit_top
        or " by " in q
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if group and value:

            return {
                "operation": (
                    "filtered_top_n"
                    if filtered
                    else "top_n"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "group_column": group,
                "value_column": value,
                "n": _extract_n(
                    question,
                    1,
                ),
            }

    # --------------------------------------------------------
    # PERCENTAGE / SHARE
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "percentage of total",
            "percent of total",
            "contribution",
            "share of",
            "share ",
        )
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if (
            group
            and value
            and not filtered
        ):
            return {
                "operation":
                    "percentage_of_total",
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "monthly",
            "by month",
            "per month",
            "month wise",
            "month-wise",
            "monthly trend",
            "monthly revenue",
            "monthly sales",
            "sales by month",
            "revenue by month",
        )
    ):

        date = _semantic_column(
            get_dataset_columns(df),
            [
                "date",
                "datetime",
                "timestamp",
            ],
        )

        value = _value_column(
            question,
            df,
        )

        if date and value:

            return {
                "operation": "monthly_sum",
                "date_column": date,
                "value_column": value,
            }

    # --------------------------------------------------------
    # UNIQUE / DISTINCT
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "unique",
            "distinct",
            "different ",
        )
    ):

        column = _unique_column(
            question,
            df,
        )

        if filtered:

            return {
                "operation":
                    "filtered_unique_count",
                "filters": filters,
                "value_column": column,
            }

        return {
            "operation":
                "calculate_unique_count",
            "column": column,
        }

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "how many",
            "count",
            "number of",
            "record count",
            "number of rows",
            "rows",
        )
    ):

        column = _count_column(
            question,
            df,
        )

        if filtered:

            return {
                "operation":
                    "filtered_count",
                "filters": filters,
                "count_column": column,
            }

        return {
            "operation":
                "calculate_count",
            "column": column,
        }

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "average",
            "avg",
            "mean",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if not value:
            raise ValueError(
                "Could not determine "
                "the average column."
            )

        if filtered:

            return {
                "operation":
                    "filtered_average",
                "filters": filters,
                "value_column": value,
            }

        if grouping_intent:

            group = _group_column(
                question,
                df,
            )

            if group:

                return {
                    "operation":
                        "group_and_average",
                    "group_column": group,
                    "value_column": value,
                }

        return {
            "operation":
                "calculate_average",
            "column": value,
        }

    # --------------------------------------------------------
    # MINIMUM
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "minimum",
            "minimum value",
            "lowest value",
            "smallest value",
            "lowest",
            "least",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if value:

            return {
                "operation": (
                    "filtered_min"
                    if filtered
                    else "calculate_min"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                **(
                    {
                        "value_column": value
                    }
                    if filtered
                    else {
                        "column": value
                    }
                ),
            }

    # --------------------------------------------------------
    # MAXIMUM
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "maximum value",
            "highest value",
            "largest value",
            "maximum",
            "highest",
            "largest",
            "greatest",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if value:

            return {
                "operation": (
                    "filtered_max"
                    if filtered
                    else "calculate_max"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                **(
                    {
                        "value_column": value
                    }
                    if filtered
                    else {
                        "column": value
                    }
                ),
            }

    # --------------------------------------------------------
    # VALUE FREQUENCY
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "frequency",
            "frequencies",
            "distribution",
            "most common",
            "value counts",
            "how often",
        )
    ):

        column = _find_column(
            question,
            get_dataset_columns(df),
        )

        column = (
            column
            or _group_column(
                question,
                df,
            )
        )

        if column:

            return {
                "operation": (
                    "filtered_value_counts"
                    if filtered
                    else "value_counts"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "column": column,
            }

    # --------------------------------------------------------
    # GROUPED OPERATIONS
    # --------------------------------------------------------

    if grouping_intent or any(
        phrase in q
        for phrase in (
            "by city",
            "by product",
            "by category",
            "by department",
            "by region",
            "by state",
            "by country",
            "group by",
            "grouped",
        )
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if group and value:

            if (
                "average" in q
                or "mean" in q
            ):

                return {
                    "operation": (
                        "filtered_group_and_average"
                        if filtered
                        else "group_and_average"
                    ),
                    **(
                        {
                            "filters": filters
                        }
                        if filtered
                        else {}
                    ),
                    "group_column": group,
                    "value_column": value,
                }

            return {
                "operation": (
                    "filtered_group_and_sum"
                    if filtered
                    else "group_and_sum"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # SUM / TOTAL
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "total",
            "sum",
            "revenue",
            "sales",
            "total revenue",
            "total sales",
            "total amount",
            "total cost",
            "total profit",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if not value:
            raise ValueError(
                "Could not determine "
                "the numeric column."
            )

        if filtered:

            return {
                "operation":
                    "filtered_sum",
                "filters": filters,
                "value_column": value,
            }

        return {
            "operation":
                "calculate_sum",
            "column": value,
        }

    raise ValueError(
        "Could not determine an analysis "
        "operation from the question."
    )


# ============================================================
# GEMINI PLANNER
# ============================================================

def _gemini_plan(
    question: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Ask Gemini to create the analysis plan."""

    prompt = f"""
Return exactly one JSON analysis plan.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

SUPPORTED OPERATIONS:
{json.dumps(
    sorted(SUPPORTED_OPERATIONS),
    indent=2,
)}

RULES:

1. Use only actual dataset columns.

2. Never invent column names.

3. Never invent filter values.

4. Filters must contain:
   column
   operator
   value

5. Equality uses "=".

6. Supported filter operators:
   =
   !=
   >
   >=
   <
   <=
   contains
   between

7. Do not calculate any result.

8. Return JSON only.

9. For top_n and filtered_top_n,
   n must be an integer.

10. For filtered operations,
    all conditions must be inside filters.

11. Return exactly one operation.
"""

    response = get_gemini_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": GeminiPlan,
        },
    )

    response_text = getattr(
        response,
        "text",
        None,
    )

    if not response_text:
        raise ValueError(
            "Gemini returned an empty analysis plan."
        )

    return extract_json(
        response_text
    )


# ============================================================
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(
    question: str,
    profile: Optional[
        Dict[str, Any]
    ],
    df: Optional[
        pd.DataFrame
    ] = None,
) -> Dict[str, Any]:
    """Create and validate an analysis plan with Gemini.

    Gemini is the primary and required planner. Python remains the
    source of truth for schema validation and all calculations.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if df is None:
        raise ValueError(
            "choose_analysis requires the real DataFrame."
        )

    normalized_profile = normalize_profile(
        df,
        profile,
    )

    try:
        plan = _gemini_plan(
            question,
            normalized_profile,
        )

        return validate_plan(
            plan,
            normalized_profile,
        )

    except Exception as exc:
        raise ValueError(
            "Gemini could not create a valid analysis plan: "
            f"{exc}"
        ) from exc


# ============================================================
# FILTERED GROUP COUNT
# ============================================================

def filtered_group_and_count(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
) -> pd.DataFrame:
    """Group and count rows after applying the supplied filters."""

    validate_dataframe(df)

    filtered_df = apply_filters(
        df,
        filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return group_and_count(
        filtered_df,
        group_column,
    )


# ============================================================
# EXECUTION
# ============================================================

def execute_analysis(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    """
    Execute a validated analysis plan.

    Python performs all actual calculations.
    """

    validate_dataframe(df)

    plan = normalize_plan(
        df,
        plan,
    )

    operation = plan[
        "operation"
    ]

    # --------------------------------------------------------
    # BASIC OPERATIONS
    # --------------------------------------------------------

    if operation == "calculate_sum":

        return calculate_sum(
            df,
            plan["column"],
        )

    if operation == "calculate_average":

        return calculate_average(
            df,
            plan["column"],
        )

    if operation == "calculate_count":

        return calculate_count(
            df,
            plan["column"],
        )

    if operation == "calculate_unique_count":

        return calculate_unique_count(
            df,
            plan["column"],
        )

    if operation == "calculate_min":

        return calculate_min(
            df,
            plan["column"],
        )

    if operation == "calculate_max":

        return calculate_max(
            df,
            plan["column"],
        )

    # --------------------------------------------------------
    # GROUP OPERATIONS
    # --------------------------------------------------------

    if operation == "group_and_sum":

        return group_and_sum(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if operation == "group_and_average":

        return group_and_average(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if operation == "group_and_count":

        return group_and_count(
            df,
            plan["group_column"],
        )

    # --------------------------------------------------------
    # TOP N
    # --------------------------------------------------------

    if operation == "top_n":

        return top_n(
            df,
            plan["group_column"],
            plan["value_column"],
            plan.get(
                "n",
                5,
            ),
        )

    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    if operation == "percentage_of_total":

        return percentage_of_total(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if operation == "monthly_sum":

        return monthly_sum(
            df,
            plan["date_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # VALUE COUNTS
    # --------------------------------------------------------

    if operation == "value_counts":

        return value_counts(
            df,
            plan["column"],
        )

    # --------------------------------------------------------
    # FILTERED SUM
    # --------------------------------------------------------

    if operation == "filtered_sum":

        return filtered_sum(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED AVERAGE
    # --------------------------------------------------------

    if operation == "filtered_average":

        return filtered_average(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED COUNT
    # --------------------------------------------------------

    if operation == "filtered_count":

        return filtered_count(
            df,
            plan["filters"],
            plan["count_column"],
        )

    # --------------------------------------------------------
    # FILTERED UNIQUE COUNT
    # --------------------------------------------------------

    if operation == "filtered_unique_count":

        return filtered_unique_count(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED MIN
    # --------------------------------------------------------

    if operation == "filtered_min":

        return filtered_min(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED MAX
    # --------------------------------------------------------

    if operation == "filtered_max":

        return filtered_max(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED GROUP COUNT
    # --------------------------------------------------------

    if operation == "filtered_group_and_count":

        return filtered_group_and_count(
            df,
            plan["filters"],
            plan["group_column"],
        )

    # --------------------------------------------------------
    # FILTERED GROUP SUM
    # --------------------------------------------------------

    if operation == "filtered_group_and_sum":

        return filtered_group_and_sum(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED GROUP AVERAGE
    # --------------------------------------------------------

    if operation == "filtered_group_and_average":

        return filtered_group_and_average(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED VALUE COUNTS
    # --------------------------------------------------------

    if operation == "filtered_value_counts":

        return filtered_value_counts(
            df,
            plan["filters"],
            plan["column"],
        )

    # --------------------------------------------------------
    # FILTERED TOP N
    # --------------------------------------------------------

    if operation == "filtered_top_n":

        return filtered_top_n(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
            plan.get(
                "n",
                5,
            ),
        )

    raise ValueError(
        f"Unsupported operation: "
        f"{operation}"
    )


# ============================================================
# GEMINI RESULT EXPLANATION
# ============================================================

def explain_result(
    question: str,
    plan: Dict[str, Any],
    result: Any,
) -> str:
    """Use Gemini to explain the actual Python result.

    Gemini may explain the result, but it is never allowed to replace
    the Python calculation or invent a value that is not present in it.
    """

    result_data = serialize_result(
        result
    )

    prompt = f"""
You are an expert data analyst.

Answer the user's question using ONLY
the ACTUAL PYTHON RESULT.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(
    plan,
    indent=2,
    default=str,
)}

PYTHON RESULT:
{json.dumps(
    result_data,
    indent=2,
    default=str,
)}

RULES:

1. Never invent numbers.

2. Use only the actual Python result.

3. Be concise but useful.

4. Format large numbers with commas.

5. Explain the key finding.

6. If the result is grouped data, identify the highest relevant group
   when that is directly evident from the supplied result.

7. If the result is a percentage table, identify the largest contribution
   when directly evident from the supplied result.

8. If the result is monthly data, identify the highest month
   when directly evident from the supplied result.

9. For unique-count operations, clearly say unique or distinct.

10. For filtered operations, mention the applied conditions.

11. Do not confuse record count with unique count.

12. Do not mention internal prompts.

13. Do not mention implementation details unless the user asks.

14. Do not perform a new calculation.

15. Return only the final natural-language answer.
"""

    try:
        response = get_gemini_client().models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            "Gemini failed while explaining the analysis result: "
            f"{exc}"
        ) from exc

    text = getattr(
        response,
        "text",
        None,
    )

    if not text or not text.strip():
        raise RuntimeError(
            "Gemini returned an empty explanation."
        )

    return text.strip()


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    profile: Optional[
        Dict[str, Any]
    ],
    question: str,
) -> Dict[str, Any]:
    """
    Complete AI Data Analyst pipeline.

    Flow:

        User Question
              |
              v
        Gemini Planner
              |
              v
        Validated Gemini plan
              |
              v
        Normalize Plan
              |
              v
        Validate Columns
              |
              v
        Python Executes
              |
              v
        Actual Result
              |
              v
        Gemini Explanation
              |
              v
        Final Response

    Important:

    Gemini chooses and explains the analysis, while Python performs
    and owns the actual calculation. Gemini never supplies the numeric result.
    """

    validate_dataframe(df)

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    normalized_profile = normalize_profile(
        df,
        profile,
    )

    # --------------------------------------------------------
    # STEP 1
    # Choose operation.
    # --------------------------------------------------------

    plan = choose_analysis(
        question,
        normalized_profile,
        df=df,
    )

    # --------------------------------------------------------
    # STEP 2
    # Normalize and validate again.
    #
    # This protects the execution boundary.
    # --------------------------------------------------------

    plan = normalize_plan(
        df,
        plan,
    )

    # --------------------------------------------------------
    # STEP 3
    # Execute actual Python calculation.
    # --------------------------------------------------------

    result = execute_analysis(
        df,
        plan,
    )

    # --------------------------------------------------------
    # STEP 4
    # Explain actual result.
    # --------------------------------------------------------

    explanation = explain_result(
        question,
        plan,
        result,
    )

    # --------------------------------------------------------
    # STEP 5
    # Return structured response.
    # --------------------------------------------------------

    return {
        "plan": plan,
        "result": serialize_result(
            result
        ),
        "explanation": explanation,
    }


# ============================================================
# UI-FRIENDLY ENTRY POINT
# ============================================================

def analyze_data(
    df: pd.DataFrame,
    question: str,
    profile: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper around run_analysis() for UI layers such as
    Streamlit's app.py.

    Differences from run_analysis():

    - profile is optional; it is built automatically via
      data_profile.profile_data() when not supplied.
    - "result" is returned as the *raw* Python object (e.g. an actual
      pandas DataFrame) rather than the JSON-serialized form, so callers
      can render tables/charts directly. A JSON-safe copy is also
      included under "result_json" for logging/display purposes.
    """

    validate_dataframe(df)

    if profile is None:
        from data_profile import profile_data

        profile = profile_data(df)

    normalized_profile = normalize_profile(
        df,
        profile,
    )

    plan = choose_analysis(
        question,
        normalized_profile,
        df=df,
    )

    plan = normalize_plan(
        df,
        plan,
    )

    result = execute_analysis(
        df,
        plan,
    )

    explanation = explain_result(
        question,
        plan,
        result,
    )

    return {
        "plan": plan,
        "result": result,
        "result_json": serialize_result(
            result
        ),
        "explanation": explanation,
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    """
    Backward-compatible alias.

    Older application code may call execute_plan().
    """

    return execute_analysis(
        df,
        plan,
    )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_df = pd.DataFrame(
        {
            "City": [
                "Delhi",
                "Delhi",
                "Mumbai",
            ],
            "Product": [
                "Laptop",
                "Phone",
                "Laptop",
            ],
            "Revenue": [
                100,
                50,
                200,
            ],
        }
    )

    test_profile = build_profile(
        test_df
    )

    test_questions = [
        "What is the total revenue?",
        "What is the total revenue in Delhi?",
        "What is the total revenue in Delhi for Laptop?",
        "What is the average revenue?",
        "How many orders are in Delhi?",
        "How many unique customers are in Delhi?",
        "What is the revenue by city?",
        "What is the highest revenue?",
        "What is the highest revenue by city?",
        "What are the top 2 cities by revenue?",
        "What is the frequency of products?",
    ]

    for test_question in test_questions:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "QUESTION:"
        )

        print(
            test_question
        )

        try:

            output = run_analysis(
                test_df,
                test_profile,
                test_question,
            )

            print(
                "\nPLAN:"
            )

            print(
                json.dumps(
                    output["plan"],
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nRESULT:"
            )

            print(
                json.dumps(
                    output["result"],
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nEXPLANATION:"
            )

            print(
                output["explanation"]
            )

        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(exc)