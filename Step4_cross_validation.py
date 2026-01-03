import pandas as pd

# 1. Load the fully merged dataset
merged = pd.read_csv("full_merged_data.csv")

# 2. Check the structure (sanity check)
print("=== MERGED DATA INFO ===")
merged.info()
print("\n=== FIRST 5 ROWS ===")
print(merged.head())

# 3. Average cost per PLANT
avg_cost_per_plant = (
    merged
    .groupby("Plant ID")["Cost_USD_per_MWh"]
    .mean()
    .reset_index()
)

print("\n=== Average cost per plant ===")
print(avg_cost_per_plant.head())

# 4. Average cost per PLANT TYPE (more important for discussion)
avg_cost_per_type = (
    merged
    .groupby("Plant Type")["Cost_USD_per_MWh"]
    .mean()
    .reset_index()
)

print("\n=== Average cost per plant type ===")
print(avg_cost_per_type)

# 5. Save results for the report
avg_cost_per_plant.to_csv("avg_cost_per_plant.csv", index=False)
avg_cost_per_type.to_csv("avg_cost_per_plant_type.csv", index=False)

print("\nStep 4 finished: cost analysis files saved")
