# src/evaluator.py

# src/evaluator.py

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def selection_error_rmse(y_true, y_pred, groups):
    """
    NEC selection metric:
    For each Demand ID group (d):
      - best_true_cost = min(true cost in group)
      - selected_true_cost = true cost of the row with min(predicted cost)
      - error(d) = best_true_cost - selected_true_cost
    Return RMSE over error(d) across groups.
    """
    errors = []

    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = (groups == g)

        true_costs = y_true[mask]
        pred_costs = y_pred[mask]

        best_true_cost = np.min(true_costs)
        selected_idx = np.argmin(pred_costs)
        selected_true_cost = true_costs[selected_idx]

        errors.append(best_true_cost - selected_true_cost)

    errors = np.array(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))


def evaluate_model(model, X, y, groups, config):
    """
    Runs Leave-One-Group-Out CV grouped by Demand ID.
    Returns:
      - cv_rmse: mean selection-error RMSE (NEC metric)
      - cv_rmse_std: std across folds
      - (optional) row_rmse_mean/std: standard regression RMSE (helpful for debugging)
    """
    logo = LeaveOneGroupOut()

    sel_rmses = []
    row_rmses = []

    # NOTE: X/y/groups should be pandas objects because you use .iloc
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        # Correct NEC metric
        sel_rmse = selection_error_rmse(
            y_true=y_test.to_numpy(),
            y_pred=np.asarray(preds),
            groups=g_test.to_numpy()
        )
        sel_rmses.append(sel_rmse)

        # Optional: standard row-level RMSE (debug / extra evidence)
        row_rmse = mean_squared_error(y_test, preds, squared=False)
        row_rmses.append(float(row_rmse))

    return {
        "cv_rmse": float(np.mean(sel_rmses)),
        "cv_rmse_std": float(np.std(sel_rmses)),
        "row_rmse_mean": float(np.mean(row_rmses)),
        "row_rmse_std": float(np.std(row_rmses)),
    }

