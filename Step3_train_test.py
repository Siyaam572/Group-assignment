# step3_merge_data.py

import pandas as pd

# Load cleaned datasets
demand = pd.read_csv("demand_clean.csv")
plants = pd.read_csv("plants_clean.csv")
generation_costs = pd.read_csv("generation_costs_clean.csv")

# Merge demand with generation costs
merged_1 = pd.merge(
    demand,
    generation_costs,
    on="Demand ID",
    how="inner"
)

# Merge with plants data
full_merged = pd.merge(
    merged_1,
    plants,
    on="Plant ID",
    how="inner"
)

# Check result
print(full_merged.info())

# Save final dataset
full_merged.to_csv("full_merged_data.csv", index=False)

print("Step 3 finished: full_merged_data.csv saved")

