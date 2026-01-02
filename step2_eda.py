# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# FILE PATHS
DATA_DIR = "prepared_data"
MERGED_FILE = f"{DATA_DIR}/merged_clean.csv"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD DATA
print("Loading cleaned datasets...")

df = pd.read_csv(MERGED_FILE)

print("Datasets loaded.")


# AUTO-DETECT KEY COLUMNS
def detect_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

DEMAND_ID_COL = detect_column(df, ["demand"])
PLANT_ID_COL = detect_column(df, ["plant"])
COST_COL = detect_column(df, ["cost"])

if DEMAND_ID_COL is None or PLANT_ID_COL is None or COST_COL is None:
    raise ValueError("Could not auto-detect ID or cost columns.")

print(f"Detected Demand ID column: {DEMAND_ID_COL}")
print(f"Detected Plant ID column: {PLANT_ID_COL}")
print(f"Detected Cost column: {COST_COL}")


# IDENTIFY FEATURE COLUMNS
demand_features = [c for c in df.columns if c.startswith("DF")]
plant_features = [c for c in df.columns if c.startswith("PF")]


# 2.1 DEMAND FEATURE ANALYSIS
print("\n2.1 Analysing demand feature distributions...")

# Summary statistics
df[demand_features].describe().T.to_csv(
    f"{OUTPUT_DIR}/demand_feature_summary.csv"
)

# Histograms
for col in demand_features:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{col}_distribution.png")
    plt.close()

# Correlation matrix
corr = df[demand_features].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(demand_features)), demand_features, rotation=90)
plt.yticks(range(len(demand_features)), demand_features)
plt.title("Demand Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/demand_feature_correlation_matrix.png")
plt.close()


# 2.2 COST DATA ANALYSIS
print("\n2.2 Analysing cost data...")

# Overall cost distribution
plt.figure()
df[COST_COL].hist(bins=50)
plt.title("Distribution of Generation Cost (USD/MWh)")
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cost_distribution.png")
plt.close()

# Average cost per plant
avg_cost_per_plant = (
    df.groupby(PLANT_ID_COL)[COST_COL]
    .mean()
    .sort_values()
)

avg_cost_per_plant.to_csv(
    f"{OUTPUT_DIR}/average_cost_per_plant.csv"
)

plt.figure(figsize=(10, 5))
avg_cost_per_plant.plot(kind="bar")
plt.title("Average Generation Cost per Plant")
plt.xlabel("Plant")
plt.ylabel("Average Cost")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/avg_cost_per_plant.png")
plt.close()

# Best plant frequency
best_plant_per_demand = (
    df.loc[
        df.groupby(DEMAND_ID_COL)[COST_COL].idxmin()
    ]
)

best_counts = best_plant_per_demand[PLANT_ID_COL].value_counts()
best_counts.to_csv(
    f"{OUTPUT_DIR}/best_plant_frequency.csv"
)

plt.figure(figsize=(10, 5))
best_counts.plot(kind="bar")
plt.title("Frequency of Being Best Plant")
plt.xlabel("Plant")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/best_plant_frequency.png")
plt.close()


# 2.3 ERROR & BASELINE RMSE
print("\n2.3 Analysing error distribution and baseline RMSE...")

# Best cost per demand
best_cost = (
    df.groupby(DEMAND_ID_COL)[COST_COL]
    .min()
    .rename("BestCost")
)

error_df = df.merge(best_cost, on=DEMAND_ID_COL)
error_df["Error"] = error_df["BestCost"] - error_df[COST_COL]

# Error distribution
plt.figure()
error_df["Error"].hist(bins=50)
plt.title("Distribution of Error(d)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/error_distribution.png")
plt.close()

# Baseline RMSE per plant
rmse_per_plant = (
    error_df
    .groupby(PLANT_ID_COL)["Error"]
    .apply(lambda e: np.sqrt(np.mean(e ** 2)))
    .sort_values()
)

rmse_per_plant.to_csv(
    f"{OUTPUT_DIR}/rmse_per_plant.csv"
)

plt.figure(figsize=(10, 5))
rmse_per_plant.plot(kind="bar")
plt.title("Baseline RMSE per Plant")
plt.xlabel("Plant")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rmse_per_plant.png")
plt.close()

print("\nSTEP 2 EDA COMPLETED SUCCESSFULLY.")
print(f"All outputs saved to '{OUTPUT_DIR}/'")
