# pip install pandas numpy matplotlib seaborn scikit-learn

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset
file_path = 'Automobile_data.csv'
df = pd.read_csv(file_path)

# Step 3: Replace '?' with NaN for better handling
df.replace('?', np.nan, inplace=True)

# Step 4: Check missing values
print("🔍 Missing values (before imputation):")
print(df.isnull().sum())

# Step 5: Convert numeric columns to proper type
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Step 6: Impute numerical columns with mean or median
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df.fillna({col: df[col].mean()}, inplace=True)

  # or use median

# Step 7: Impute categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df.fillna({col: df[col].mode()[0]}, inplace=True)


# Step 8: Check again
print("\n✅ Missing values (after imputation):")
print(df.isnull().sum())

# Step 9: Preview the cleaned dataset
print("\n📋 Cleaned data after imputation:\n", df.head())
