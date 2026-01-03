# step2_eda.py
# Step 2: Exploratory Data Analysis (EDA)

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load cleaned datasets
demand = pd.read_csv("demand_clean.csv")
plants = pd.read_csv("plants_clean.csv")
generation_costs = pd.read_csv("generation_costs_clean.csv")

# 3. Separate numerical and categorical columns
demand_num = demand.select_dtypes(include=["number"])
demand_cat = demand.select_dtypes(include=["object"])

plants_num = plants.select_dtypes(include=["number"])
plants_cat = plants.select_dtypes(include=["object"])

generation_costs_num = generation_costs.select_dtypes(include=["number"])

# 4. Summary statistics for numerical data
print("=== DEMAND NUMERICAL SUMMARY ===")
print(demand_num.describe())

print("\n=== PLANTS NUMERICAL SUMMARY ===")
print(plants_num.describe())

print("\n=== GENERATION COSTS SUMMARY ===")
print(generation_costs_num.describe())

# 5. Frequency counts for categorical variables
print("\n=== DEMAND CATEGORICAL COUNTS ===")
for col in demand_cat.columns:
    print(f"\n{col}")
    print(demand[col].value_counts())

print("\n=== PLANTS CATEGORICAL COUNTS ===")
for col in plants_cat.columns:
    print(f"\n{col}")
    print(plants[col].value_counts())

# 6. Simple visualisations

# Histogram of one demand feature
plt.hist(demand["DF1"], bins=20)
plt.title("Distribution of DF1")
plt.xlabel("DF1 value")
plt.ylabel("Frequency")
plt.show()

# Histogram of generation costs
plt.hist(generation_costs["Cost_USD_per_MWh"], bins=30)
plt.title("Distribution of Generation Costs")
plt.xlabel("Cost (USD per MWh)")
plt.ylabel("Frequency")
plt.show()

print("\nStep 2 finished: EDA completed")
