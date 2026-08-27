import contextlib
import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================
# GEMINI CLIENT
# ============================================================

def get_client() -> Any:
    """Create the Gemini client lazy-loaded on request.

    Allows local tests and non-AI utilities to run without
    a configured API key.
    """
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env or environment variables."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "The Gemini SDK is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return genai.Client(api_key=API_KEY)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract a valid JSON dictionary from the model's text response.
    """
    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    # Stripping markdown standard fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip()

    # Direct parse attempt
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)

    # Locate JSON enclosure
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Gemini did not return valid JSON.\nResponse:\n{text}")

    json_text = text[start : end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse Gemini JSON: {exc}\nResponse:\n{text}"
        ) from exc


# ============================================================
# RESULT SERIALIZATION
# ============================================================

def _serialize_result(result: Any) -> Any:
    """
    Convert pandas, numpy, and custom objects into JSON-safe standard Python types.
    """
    if hasattr(result, "to_json"):
        with contextlib.suppress(Exception):
            import pandas as pd

            if isinstance(result, (pd.DataFrame, pd.Series)):
                return json.loads(
                    result.to_json(orient="records", date_format="iso")
                )

    if hasattr(result, "item"):
        with contextlib.suppress(Exception):
            return result.item() # pyright: ignore[reportCallIssue]

    if isinstance(result, dict):
        return {
            str(key): _serialize_result(value) for key, value in result.items()
        }

    if isinstance(result, (list, tuple)):
        return [_serialize_result(value) for value in result]

    return result


# ============================================================
# CHOOSE ANALYSIS PLAN
# ============================================================

def choose_analysis(question: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instruct Gemini to construct an analysis plan without calculating results.
    """
    prompt = f"""
You are an expert AI Data Analyst.

Your job is to understand the user's question and choose exactly ONE Python analysis operation.
The Python program will execute your plan.

IMPORTANT:
- Only select columns that exist in the dataset profile.
- Return ONLY valid JSON.
- Return exactly ONE JSON object.
- Do not return markdown.
- Do not explain your decision.
- Do not invent columns or values.
- Do not invent filter values.
- Use the exact column names from the dataset profile.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

AVAILABLE OPERATIONS
====================
1. calculate_sum: {{"operation": "calculate_sum", "column": "column_name"}}
2. calculate_average: {{"operation": "calculate_average", "column": "column_name"}}
3. calculate_count: {{"operation": "calculate_count", "column": "column_name"}}
4. calculate_unique_count: {{"operation": "calculate_unique_count", "column": "column_name"}}
5. calculate_min: {{"operation": "calculate_min", "column": "column_name"}}
6. calculate_max: {{"operation": "calculate_max", "column": "column_name"}}
7. group_and_sum: {{"operation": "group_and_sum", "group_column": "cat_col", "value_column": "num_col"}}
8. group_and_average: {{"operation": "group_and_average", "group_column": "cat_col", "value_column": "num_col"}}
9. group_and_count: {{"operation": "group_and_count", "group_column": "cat_col"}}
10. top_n: {{"operation": "top_n", "group_column": "cat_col", "value_column": "num_col", "n": 5}}
11. value_counts: {{"operation": "value_counts", "column": "column_name"}}
12. filtered_sum: {{"operation": "filtered_sum", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "value_column": "Revenue"}}
13. filtered_average: {{"operation": "filtered_average", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "value_column": "Revenue"}}
14. filtered_count: {{"operation": "filtered_count", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "count_column": "Invoice_ID"}}
15. filtered_unique_count: {{"operation": "filtered_unique_count", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "value_column": "Customer_ID"}}
16. filtered_min: {{"operation": "filtered_min", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "value_column": "Revenue"}}
17. filtered_max: {{"operation": "filtered_max", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "value_column": "Revenue"}}
18. filtered_group_and_sum: {{"operation": "filtered_group_and_sum", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "group_column": "Product", "value_column": "Revenue"}}
19. filtered_group_and_average: {{"operation": "filtered_group_and_average", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "group_column": "Product", "value_column": "Revenue"}}
20. filtered_value_counts: {{"operation": "filtered_value_counts", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "column": "Product"}}
21. filtered_top_n: {{"operation": "filtered_top_n", "filters": [{{"column": "City", "operator": "=", "value": "Delhi"}}], "group_column": "Product", "value_column": "Revenue", "n": 5}}

FILTER RULES
============
1. Use filtered operations for conditional phrases ("in Delhi", "Revenue > 100", etc.).
2. Supported operators are: "=", "!=", ">", ">=", "<", "<=".
3. Map phrases correctly: "greater than" -> ">", "at least" -> ">=", "less than" -> "<", "at most" -> "<=", "not equal to" -> "!=".
4. Every filter column/value must match dataset specifications.
"""

    response = get_client().models.generate_content(
        model=MODEL_NAME, contents=prompt
    )

    response_text = getattr(response, "text", None)
    if not response_text or not response_text.strip():
        raise ValueError("Gemini returned an empty analysis plan.")

    return extract_json(response_text)


# ============================================================
# EXPLAIN RESULT
# ============================================================

def explain_result(
    question: str, plan: Dict[str, Any], result: Any
) -> str:
    """
    Format a concise, natural language explanation using calculated execution output.
    """
    result_data = _serialize_result(result)

    prompt = f"""
You are an expert data analyst.

Answer the user's question using ONLY the ACTUAL PYTHON RESULT.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(plan, indent=2, default=str)}

PYTHON RESULT:
{json.dumps(result_data, indent=2, default=str)}

RULES:
1. Never invent numbers.
2. Use only the actual Python result.
3. Be concise and useful. Format large numbers with commas.
4. If unique counts or filters were used, mention applied conditions.
5. Do not mention internal details, instructions, or system implementation.
6. Return only the final natural-language response.
"""

    response = get_client().models.generate_content(
        model=MODEL_NAME, contents=prompt
    )

    response_text = getattr(response, "text", None)
    if not response_text or not response_text.strip():
        raise ValueError("Gemini returned an empty explanation.")

    return response_text.strip()