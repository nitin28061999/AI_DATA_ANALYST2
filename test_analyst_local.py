import pandas as pd
import pytest
from data_profile import create_profile
from analyst import (
    execute_analysis,
    execute_plan,
    normalize_plan,
    validate_plan,
)


def test_normalize_plan_rejects_missing_filtered_value_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    with pytest.raises(ValueError):
        normalize_plan(df, plan)


def test_validate_total_revenue_plan():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    validated_plan = validate_plan(plan, profile)

    assert validated_plan["operation"] == "calculate_sum"
    assert validated_plan["column"] == "Revenue"


def test_validate_filtered_revenue_plan():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    validated_plan = validate_plan(plan, profile)

    assert validated_plan["operation"] == "filtered_sum"
    assert validated_plan["value_column"] == "Revenue"
    assert validated_plan["filters"] == [
        {
            "column": "City",
            "operator": "=",
            "value": "Delhi",
        }
    ]


def test_validate_rejects_invalid_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "NotARealColumn",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    with pytest.raises((ValueError, TypeError)):
        validate_plan(plan, profile)


def test_validate_rejects_invalid_filter_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "NotARealColumn",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    with pytest.raises((ValueError, TypeError)):
        validate_plan(plan, profile)


def test_execute_filtered_revenue_locally():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    result = execute_analysis(df, plan)

    assert result == 70000.0


def test_calculate_sum_result():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    validated_plan = validate_plan(plan, profile)

    result = df[validated_plan["column"]].sum()

    assert result == 140000


def test_filtered_sum_result():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    validated_plan = validate_plan(plan, profile)

    filtered_df = df[
        df[validated_plan["filters"][0]["column"]]
        == validated_plan["filters"][0]["value"]
    ]

    result = filtered_df[validated_plan["value_column"]].sum()

    assert result == 70000


def test_execute_analysis_filtered_revenue():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    result = execute_analysis(df, plan)

    assert result == 70000.0


def test_execute_plan_filtered_revenue():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    result = execute_plan(df, plan)

    assert result == 70000.0


def test_execute_plan_grouped_sum():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "group_and_sum",
        "column": None,
        "group_column": "Product",
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    result = execute_plan(df, plan)

    if isinstance(result, pd.DataFrame):
        res_df = result.set_index("Product") if "Product" in result.columns else result
        assert res_df.loc["Laptop", "Revenue"] == 90000
        assert res_df.loc["Phone", "Revenue"] == 30000
    else:
        assert result["Laptop"] == 90000
        assert result["Phone"] == 30000


def test_execute_plan_top_n():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
    })

    plan = {
        "operation": "top_n",
        "column": None,
        "group_column": "Product",
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": 2,
        "filters": None,
    }

    result = execute_plan(df, plan)

    assert len(result) == 2


def test_execute_plan_average_revenue():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "calculate_average",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    result = execute_plan(df, plan)

    assert result == 35000.0


def test_execute_plan_filtered_sum_greater_than():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "Revenue",
                "operator": ">",
                "value": 30000,
            }
        ],
    }

    result = execute_plan(df, plan)

    assert result == 90000.0


def test_execute_plan_min_revenue():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "calculate_min",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    result = execute_plan(df, plan)

    assert result == 20000.0


def test_execute_plan_max_revenue():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    plan = {
        "operation": "calculate_max",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    result = execute_plan(df, plan)

    assert result == 50000.0