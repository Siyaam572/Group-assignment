import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# FILE PATHS
DEMAND_FILE = "demand.csv"
PLANTS_FILE = "plants.csv"
COST_FILE = "generation_costs.csv"

OUTPUT_DIR = "prepared_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K_PLANTS = 40
VAR_THRESHOLD = 0.01


# LOAD DATASETS

print("Loading datasets...")

demand_df = pd.read_csv(DEMAND_FILE)
plants_df = pd.read_csv(PLANTS_FILE)
costs_df = pd.read_csv(COST_FILE)

print("Data loaded successfully.")


# AUTO-DETECT KEY COLUMNS

def detect_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

DEMAND_ID_COL = detect_column(demand_df, ["demand"])
PLANT_ID_COL = detect_column(plants_df, ["plant"])
COST_COL = detect_column(costs_df, ["cost"])

if DEMAND_ID_COL is None:
    raise ValueError("Could not detect Demand ID column in demand.csv")

if PLANT_ID_COL is None:
    raise ValueError("Could not detect Plant ID column in plants.csv")

if COST_COL is None:
    raise ValueError("Could not detect Cost column in generation_costs.csv")

print(f"Detected Demand ID column: {DEMAND_ID_COL}")
print(f"Detected Plant ID column: {PLANT_ID_COL}")
print(f"Detected Cost column: {COST_COL}")


# HANDLE MISSING VALUES

def handle_missing(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

print("\nHandling missing values...")
demand_df = handle_missing(demand_df)
plants_df = handle_missing(plants_df)
costs_df = handle_missing(costs_df)
print("Missing values handled.")


# IDENTIFY NUMERIC FEATURES

print("\nIdentifying numeric features...")

demand_numeric = demand_df.select_dtypes(include=[np.number]).columns.tolist()
plant_numeric = plants_df.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric demand features:", demand_numeric)
print("Numeric plant features:", plant_numeric)


# FEATURE SELECTION

print("\nApplying variance threshold feature selection...")

selector = VarianceThreshold(threshold=VAR_THRESHOLD)

selector.fit(demand_df[demand_numeric])
selected_demand_features = [
    demand_numeric[i] for i, v in enumerate(selector.variances_) if v > VAR_THRESHOLD
]

selector.fit(plants_df[plant_numeric])
selected_plant_features = [
    plant_numeric[i] for i, v in enumerate(selector.variances_) if v > VAR_THRESHOLD
]

print("Selected demand features:", selected_demand_features)
print("Selected plant features:", selected_plant_features)

# Reduce datasets
demand_df = demand_df[[DEMAND_ID_COL] + selected_demand_features]
plants_df = plants_df[[PLANT_ID_COL] + selected_plant_features]


# FEATURE SCALING

print("\nScaling numeric features...")

scaler = StandardScaler()

demand_df[selected_demand_features] = scaler.fit_transform(
    demand_df[selected_demand_features]
)

plants_df[selected_plant_features] = scaler.fit_transform(
    plants_df[selected_plant_features]
)

print("Feature scaling completed.")


# PLANT PERFORMANCE ANALYSIS

print("\nAnalysing plant performance...")

best_cost = (
    costs_df.groupby(DEMAND_ID_COL)[COST_COL]
    .min()
    .rename("BestCost")
)

costs_eval = costs_df.merge(best_cost, on=DEMAND_ID_COL)
costs_eval["Error"] = costs_eval["BestCost"] - costs_eval[COST_COL]

plant_rmse = (
    costs_eval
    .groupby(PLANT_ID_COL)["Error"]
    .apply(lambda x: np.sqrt(np.mean(x**2)))
    .sort_values()
)

print("\nBaseline RMSE per plant:")
print(plant_rmse)


# PLANT PRUNING

kept_plants = plant_rmse.head(TOP_K_PLANTS).index.tolist()

plants_df = plants_df[plants_df[PLANT_ID_COL].isin(kept_plants)]
costs_df = costs_df[costs_df[PLANT_ID_COL].isin(kept_plants)]

print(f"\nKept top {TOP_K_PLANTS} plants.")


# FINAL MERGE

print("\nCreating merged dataset...")

merged_df = (
    costs_df
    .merge(demand_df, on=DEMAND_ID_COL, how="inner")
    .merge(plants_df, on=PLANT_ID_COL, how="inner")
)

print("Final merged dataset shape:", merged_df.shape)


# SAVE OUTPUT FILES
print("\nSaving prepared datasets...")
demand_df.to_csv(f"{OUTPUT_DIR}/demand_clean.csv", index=False)
plants_df.to_csv(f"{OUTPUT_DIR}/plants_clean.csv", index=False)
costs_df.to_csv(f"{OUTPUT_DIR}/costs_clean.csv", index=False)
merged_df.to_csv(f"{OUTPUT_DIR}/merged_clean.csv", index=False)

print("\nSTEP 1 COMPLETED SUCCESSFULLY.")
