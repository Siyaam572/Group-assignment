import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the merged dataset (USE THE CORRECT FILE NAME)
merged = pd.read_csv("full_merged_data.csv")

# 2. Calculate average cost per plant type
avg_cost_per_type = (
    merged
    .groupby("Plant Type")["Cost_USD_per_MWh"]
    .mean()
    .reset_index()
)

print("=== Average cost per plant type ===")
print(avg_cost_per_type)

# 3. Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(
    avg_cost_per_type["Plant Type"],
    avg_cost_per_type["Cost_USD_per_MWh"]
)
plt.title("Average Generation Cost per Plant Type")
plt.xlabel("Plant Type")
plt.ylabel("Cost (USD per MWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Save results
avg_cost_per_type.to_csv("avg_cost_per_plant_type.csv", index=False)

print("Step 5 finished: avg_cost_per_plant_type.csv saved")
