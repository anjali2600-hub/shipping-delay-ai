"""
explore_amazon.py
Quick look at the raw Amazon delivery dataset: size, columns,
missing values, and the values of each categorical column.
"""

import glob
import pandas as pd

files = glob.glob("data/raw/*.csv")
print("CSV files found in data/raw:", files)

if not files:
    raise SystemExit("No CSV found in data/raw. Copy the Kaggle CSV there first.")

df = pd.read_csv(files[0])

print("\nShape (rows, columns):", df.shape)

print("\nColumns and data types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nCategorical columns - unique values:")
for col in df.select_dtypes(include="object").columns:
    n_unique = df[col].nunique()
    if n_unique <= 15:
        print(f"  {col}: {sorted(df[col].dropna().astype(str).unique())}")
    else:
        print(f"  {col}: {n_unique} unique values")

print("\nNumeric summary:")
print(df.describe().to_string())