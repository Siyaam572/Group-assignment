# src/evaluator.py

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

def evaluate_model(model, X, y, groups, config):
    """
    Runs Leave-One-Group-Out CV (grouped by Demand ID).
    Returns dict with cv_rmse.
    """

    logo = LeaveOneGroupOut()
    rmses = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        rmses.append(rmse)

    return {
        "cv_rmse": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses))
    }
