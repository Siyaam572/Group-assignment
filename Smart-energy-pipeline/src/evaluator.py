# src/evaluator.py

# src/evaluator.py

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone

def selection_error_rmse(df_fold, y_pred, group_col="Demand ID", cost_col="Cost_USD_per_MWh"):
    """
    NEC metric:
    For each Demand ID (scenario), choose the plant with MIN predicted cost.
    Compare its TRUE cost to the TRUE best cost for that demand.

    error(d) = selected_true_cost - best_true_cost
    RMSE = sqrt(mean(error(d)^2))
    """
    errors = []

    # df_fold must contain Demand ID + true Cost + predictions aligned row-wise
    for d, g in df_fold.groupby(group_col):
        true_costs = g[cost_col].to_numpy()
        pred_costs = y_pred[g.index]  # predictions aligned to df_fold index

        best_true = true_costs.min()
        selected_idx = np.argmin(pred_costs)          # plant chosen by the model
        selected_true = true_costs[selected_idx]      # true cost of chosen plant

        error_d = selected_true - best_true
        errors.append(error_d)

    errors = np.array(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))


def grouped_train_test_split(df, group_col="Demand ID", test_size=0.2, random_state=42):
    """
    Split by Demand ID so the same demand scenario never appears in both train and test.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = df[group_col].unique()
    n_test = max(1, int(len(unique_groups) * test_size))
    test_groups = rng.choice(unique_groups, size=n_test, replace=False)

    test_mask = df[group_col].isin(test_groups)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    return train_df, test_df


def evaluate_model(model, X, y, groups, config, df_full=None, plant_ids=None):
    """
    Runs:
      - grouped train/test (by Demand ID)
      - LOGO CV (by Demand ID)
    Using NEC selection-error RMSE.
    
    IMPORTANT:
      We need a dataframe with Demand ID + Cost + Plant ID rows to compute the selection rule.
      The cleanest way is to pass df_full from preprocessing (recommended).
    """

    
    if df_full is None:
        # fallback: rebuild minimal frame from inputs
        df_full = pd.DataFrame({
            "Demand ID": groups.values if hasattr(groups, "values") else groups,
            "Cost_USD_per_MWh": y.values if hasattr(y, "values") else y,
        })

    group_col = "Demand ID"
    cost_col = "Cost_USD_per_MWh"

   
    # A) Grouped train/test
 
    train_df, test_df = grouped_train_test_split(
        df_full, group_col=group_col,
        test_size=config.get("evaluation", {}).get("test_size", 0.2),
        random_state=config.get("evaluation", {}).get("random_state", 42),
    )

    # Align X/y with the split
    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]
    X_test  = X.loc[test_df.index]
    y_test  = y.loc[test_df.index]

    m = clone(model)
    m.fit(X_train, y_train)
    y_pred_test = m.predict(X_test)

    test_metric = selection_error_rmse(
        df_fold=test_df,
        y_pred=pd.Series(y_pred_test, index=X_test.index),
        group_col=group_col,
        cost_col=cost_col
    )

    # B) LOGO CV (Leave-One-Group-Out)
    logo = LeaveOneGroupOut()
    fold_scores = []

    # Need numpy arrays of indices for split, but we will index into DataFrames by iloc positions
    X_reset = X.reset_index(drop=False)  # keep original index
    original_index = X_reset["index"].values

    y_reset = y.reset_index(drop=True)
    groups_reset = pd.Series(groups.values if hasattr(groups, "values") else groups).reset_index(drop=True)

    for train_idx, test_idx in logo.split(X_reset, y_reset, groups=groups_reset):
        train_original_idx = original_index[train_idx]
        test_original_idx = original_index[test_idx]

        X_tr = X.loc[train_original_idx]
        y_tr = y.loc[train_original_idx]
        X_te = X.loc[test_original_idx]

        fold_df = df_full.loc[test_original_idx].copy()

        m_fold = clone(model)
        m_fold.fit(X_tr, y_tr)
        y_pred_fold = m_fold.predict(X_te)

        fold_rmse = selection_error_rmse(
            df_fold=fold_df,
            y_pred=pd.Series(y_pred_fold, index=X_te.index),
            group_col=group_col,
            cost_col=cost_col
        )
        fold_scores.append(fold_rmse)

    fold_scores = np.array(fold_scores, dtype=float)

    return {
        "test_selection_rmse": float(test_metric),
        "cv_selection_rmse_mean": float(fold_scores.mean()),
        "cv_selection_rmse_std": float(fold_scores.std()),
        "cv_folds": int(len(fold_scores)),
    }
