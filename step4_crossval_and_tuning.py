# IMPORTS

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


# CONFIGURATION
RANDOM_STATE = 42
N_LOGO_DEMANDS = 100 # Number of demands to subsample for fast LOGO
DATA_FILE = "prepared_data/merged_clean.csv"


# LOAD DATA

print("Loading merged dataset...")
df = pd.read_csv(DATA_FILE)
print("Dataset loaded.")


# AUTO-DETECT COLUMNS
def detect_column(df, keywords):
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None

DEMAND_ID_COL = detect_column(df, ["demand"])
PLANT_ID_COL = detect_column(df, ["plant"])
COST_COL = detect_column(df, ["cost"])

if not DEMAND_ID_COL or not PLANT_ID_COL or not COST_COL:
    raise ValueError("Could not detect Demand / Plant / Cost columns")

print(f"Demand column: {DEMAND_ID_COL}")
print(f"Plant column: {PLANT_ID_COL}")
print(f"Cost column: {COST_COL}")


# FEATURES / TARGET / GROUPS

feature_cols = [c for c in df.columns if c.startswith("DF") or c.startswith("PF")]

X = df[feature_cols].values
y = df[COST_COL].values
groups = df[DEMAND_ID_COL].values
plant_ids = df[PLANT_ID_COL].values


# SUBSAMPLE DEMANDS 

rng = np.random.default_rng(RANDOM_STATE)
unique_demands = np.unique(groups)

selected_demands = rng.choice(
    unique_demands,
    size=N_LOGO_DEMANDS,
    replace=False
)

mask = np.isin(groups, selected_demands)

X = X[mask]
y = y[mask]
groups = groups[mask]
plant_ids = plant_ids[mask]

print(f"\nUsing {len(np.unique(groups))} Demand IDs for fast LOGO")


# Eq. (1) + Eq. (2)

def error_rmse_eq12(y_true, y_pred, demand_ids):
    errors = []

    for d in np.unique(demand_ids):
        mask = demand_ids == d

        true_costs = y_true[mask]
        pred_costs = y_pred[mask]

        best_true_cost = true_costs.min()
        selected_idx = np.argmin(pred_costs)
        selected_true_cost = true_costs[selected_idx]

        errors.append(best_true_cost - selected_true_cost)

    errors = np.array(errors)
    return np.sqrt(np.mean(errors ** 2))


# 4. FAST LOGO CV

print("\nRunning FAST Leave-One-Demand-Out CV...")

logo = LeaveOneGroupOut()
base_model = RandomForestRegressor(
    n_estimators=100,       
    random_state=RANDOM_STATE,
    n_jobs=-1
)

logo_scores = []

fold = 1
for train_idx, test_idx in logo.split(X, y, groups=groups):
    print(f"LOGO fold {fold}/{len(np.unique(groups))}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    demand_test = groups[test_idx]

    base_model.fit(X_train, y_train)
    y_pred = base_model.predict(X_test)

    rmse = error_rmse_eq12(y_test, y_pred, demand_test)
    logo_scores.append(rmse)

    fold += 1

logo_scores = np.array(logo_scores)

print("\nLOGO RMSE scores:")
print(logo_scores)
print(f"\nMean LOGO RMSE: {logo_scores.mean():.4f}")
print(f"Std LOGO RMSE: {logo_scores.std():.4f}")


# 5. GRID SEARCH 

print("\nStarting GridSearch with  LOGO...")

# Simpler scorer for GridSearch (row-level RMSE)
rmse_scorer = make_scorer(
    mean_squared_error,
    greater_is_better=False,
    squared=False
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=logo.split(X, y, groups=groups),
    n_jobs=-1,
    verbose=2
)

grid.fit(X, y)

print("\nBest hyperparameters:")
print(grid.best_params_)

best_model = grid.best_estimator_


# EVALUATE BEST MODEL WITH Eq. (1)

print("\nEvaluating best model using FAST LOGO...")

best_scores = []

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    demand_test = groups[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    rmse = error_rmse_eq12(y_test, y_pred, demand_test)
    best_scores.append(rmse)

best_scores = np.array(best_scores)

print("\nBest-model LOGO RMSE scores:")
print(best_scores)
print(f"\nBest-model Mean LOGO RMSE: {best_scores.mean():.4f}")
print(f"Best-model Std LOGO RMSE: {best_scores.std():.4f}")

print("\nSTEP 4 & 5 COMPLETED SUCCESSFULLY.")
