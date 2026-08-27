import pandas as pd
from analysis_tools import calculate_average, calculate_count, calculate_sum

# Sample data initialization
df = pd.DataFrame(
    {
        "Region": ["North", "South", "East", "West"],
        "Sales": [50000.0, 30000.0, 20000.0, 40000.0],
    }
)

print("SUM:")
print(calculate_sum(df, "Sales"))

print("\nAVERAGE:")
print(calculate_average(df, "Sales"))

print("\nCOUNT:")
# Fixed: added the required 'column' argument
print(calculate_count(df, "Sales"))