# step1_preparation.py

# 1. import the libraries I need
import numpy as np
import pandas as pd

# 2. set the file names for the three datasets
# (these should match the actual filenames in my folder)
demand_path = "demand.csv"
plants_path = "plants.csv"
generation_costs_path = "generation_costs.csv"

# 3. load the csv files into pandas dataframes
demand = pd.read_csv(demand_path)
plants = pd.read_csv(plants_path)
generation_costs = pd.read_csv(generation_costs_path)

# 4. print basic info to check columns and data types
print("=== DEMAND INFO ===")
demand.info()
print("\n=== PLANTS INFO ===")
plants.info()
print("\n=== GENERATION COSTS INFO ===")
generation_costs.info()

# 5. define a small helper function to check for missing data in a dataframe
def check_missing(df, name):
    """Prints whether there are missing values and how many per column."""
    print(f"\nMissing values in {name}:")
    missing_counts = df.isna().sum()          # count missing values per column
    print(missing_counts)
    # quick message for me
    if missing_counts.sum() == 0:
        print("No missing values")
    else:
        print("There ARE missing values")

# 6. use my helper to check missing values in each dataset BEFORE cleaning
check_missing(demand, "demand (before)")
check_missing(plants, "plants (before)")
check_missing(generation_costs, "generation_costs (before)")

# 7. handle missing values
# for numeric columns -> fill with the column mean
# for categorical columns -> fill with "Unknown"

def clean_missing(df):
    """Fill missing values: numeric -> mean, categorical -> 'Unknown'."""
    # numeric columns (int, float)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    # categorical columns (object)
    cat_cols = df.select_dtypes(include=["object"]).columns

    # fill numeric columns with their mean
    for col in numeric_cols:
        if df[col].isna().any():  # only do it if there are missing values
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)

    # fill categorical columns with 'Unknown'
    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna("Unknown", inplace=True)

    return df

# clean each dataframe
demand = clean_missing(demand)
plants = clean_missing(plants)
generation_costs = clean_missing(generation_costs)

# 8. check missing values again after cleaning
check_missing(demand, "demand (after)")
check_missing(plants, "plants (after)")
check_missing(generation_costs, "generation_costs (after)")

# 9. save the cleaned dataframes to new csv files
# I will use index=False so pandas does not write the DataFrame index as a column
demand.to_csv("demand_clean.csv", index=False)
plants.to_csv("plants_clean.csv", index=False)
generation_costs.to_csv("generation_costs_clean.csv", index=False)

print("\nStep 1 finished: cleaned files saved as:")
print(" - demand_clean.csv")
print(" - plants_clean.csv")
print(" - generation_costs_clean.csv")
