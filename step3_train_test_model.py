# IMPORTS
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# LOAD DATA
print("Loading merged dataset...")

DATA_FILE = "prepared_data/merged_clean.csv"
df = pd.read_csv(DATA_FILE)

print("Dataset loaded.")


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

if not DEMAND_ID_COL or not PLANT_ID_COL or not COST_COL:
    raise ValueError("Could not detect Demand / Plant / Cost columns.")

print(f"Detected Demand ID column: {DEMAND_ID_COL}")
print(f"Detected Plant ID column: {PLANT_ID_COL}")
print(f"Detected Cost column: {COST_COL}")


# IDENTIFY FEATURE COLUMNS
feature_cols = [c for c in df.columns if c.startswith("DF") or c.startswith("PF")]

if len(feature_cols) == 0:
    raise ValueError("No feature columns (DF*, PF*) found.")


# SPLIT X, y, GROUPS
X = df[feature_cols].values
y = df[COST_COL].values
groups = df[DEMAND_ID_COL].values
plant_ids = df[PLANT_ID_COL].values


# 3.2 GROUPED TRAIN / TEST SPLIT
print("\nPerforming grouped train/test split...")

rng = np.random.default_rng(42)
unique_demands = np.unique(groups)

TEST_GROUP_SIZE = 20
test_demands = rng.choice(unique_demands, size=TEST_GROUP_SIZE, replace=False)

test_mask = np.isin(groups, test_demands)

X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = y[~test_mask], y[test_mask]
groups_train, groups_test = groups[~test_mask], groups[test_mask]
plant_ids_test = plant_ids[test_mask]

print(f"Train rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Unique test demands: {len(np.unique(groups_test))}")


# 3.3 TRAIN MODEL
print("\nTraining RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Standard regression score (R²)
r2_score = model.score(X_test, y_test)
print(f"Test R² score: {r2_score:.4f}")


# 3.4 ERROR(d) CALCULATION (Eq. 1)
print("\nComputing Error(d) for each test demand...")

y_pred = model.predict(X_test)

error_results = []

for d in np.unique(groups_test):
    mask = groups_test == d

    true_costs = y_test[mask]
    predicted_costs = y_pred[mask]

    # Best possible cost (oracle)
    best_true_cost = true_costs.min()

    # Plant selected by ML (min predicted cost)
    selected_idx = np.argmin(predicted_costs)
    selected_true_cost = true_costs[selected_idx]

    error_d = best_true_cost - selected_true_cost
    error_results.append(error_d)

error_results = np.array(error_results)


# 3.4 RMSE (Eq. 2)
rmse_error = np.sqrt(np.mean(error_results ** 2))

print(f"\nRMSE over Error(d) [Eq. 2]: {rmse_error:.4f}")

# 3.5 SAVE RESULTS
results_df = pd.DataFrame({
    "Demand_ID": np.unique(groups_test),
    "Error_d": error_results
})

results_df.to_csv("step3_error_by_demand.csv", index=False)

print("\nPer-demand errors saved to step3_error_by_demand.csv")
print("\nSTEP 3 COMPLETED SUCCESSFULLY.")
