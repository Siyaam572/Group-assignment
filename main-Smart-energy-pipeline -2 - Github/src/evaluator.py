import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, make_scorer


def custom_error_metric(y_true, y_pred, groups):
    # Calculate Error(d) for each demand using Equation 1
    # Error(d) = min{c(p,d)} - c(p_ml, d)
    errors = []
    unique_demands = np.unique(groups)
    
    for demand_id in unique_demands:
        mask = groups == demand_id
        true_costs = y_true[mask]
        pred_costs = y_pred[mask]
        
        best_actual_cost = true_costs.min()
        selected_plant_idx = np.argmin(pred_costs)
        selected_actual_cost = true_costs[selected_plant_idx]
        
        error_d = best_actual_cost - selected_actual_cost
        errors.append(error_d)
    
    return np.array(errors)


def calculate_rmse_score(errors):
    # RMSE using Equation 2
    D = len(errors)
    rmse = np.sqrt((1/D) * np.sum(errors**2))
    return rmse


def evaluate_on_test_set(pipeline, X_train, X_test, y_train, y_test, groups_test):
    """Train pipeline and evaluate on test set"""
    print("\nTraining pipeline on train set")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on test set")
    y_pred = pipeline.predict(X_test)
    
    r2 = pipeline.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    
    errors = custom_error_metric(y_test, y_pred, groups_test)
    custom_rmse = calculate_rmse_score(errors)
    
    results = {
        'r2_score': r2,
        'mse': mse,
        'custom_rmse': custom_rmse,
        'mean_error': errors.mean(),
        'num_test_demands': len(np.unique(groups_test)),
        'num_optimal_selections': (errors == 0).sum()
    }
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Custom RMSE: {custom_rmse:.4f}")
    print(f"Mean Error: {errors.mean():.4f}")
    print(f"Optimal selections: {results['num_optimal_selections']}/{results['num_test_demands']}")
    
    return results


def grouped_train_test_split(X, y, groups, test_size=20, random_state=42):
    # Split data by Demand ID to prevent leakage
    print(f"\nPerforming grouped train/test split")
    print(f"Test size: {test_size} unique demands")
    
    rng = np.random.default_rng(random_state)
    unique_demands = np.unique(groups)
    
    test_demands = rng.choice(unique_demands, size=test_size, replace=False)
    test_mask = np.isin(groups, test_demands)
    
    X_train = X[~test_mask]
    X_test = X[test_mask]
    y_train = y[~test_mask]
    y_test = y[test_mask]
    groups_train = groups[~test_mask]
    groups_test = groups[test_mask]
    
    print(f"Train: {len(X_train)} rows, {len(np.unique(groups_train))} demands")
    print(f"Test: {len(X_test)} rows, {len(np.unique(groups_test))} demands")
    
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def logo_cross_validation(pipeline, X, y, groups, n_jobs=-1):
    """Perform Leave-One-Group-Out cross-validation"""
    print("\nStarting LOGO cross-validation")
    
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    print(f"Number of CV folds: {n_splits}")
    
    fold_scores = []
    fold_num = 0
    
    for train_idx, test_idx in logo.split(X, y, groups):
        fold_num += 1
        
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        groups_test_fold = groups[test_idx]
        
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred_fold = pipeline.predict(X_test_fold)
        
        errors = custom_error_metric(y_test_fold, y_pred_fold, groups_test_fold)
        rmse = calculate_rmse_score(errors)
        fold_scores.append(rmse)
        
        if fold_num % 50 == 0:
            print(f"Completed {fold_num}/{n_splits} folds")
    
    fold_scores = np.array(fold_scores)
    
    results = {
        'mean_rmse': fold_scores.mean(),
        'std_rmse': fold_scores.std(),
        'fold_scores': fold_scores,
        'min_rmse': fold_scores.min(),
        'max_rmse': fold_scores.max()
    }
    
    print("\nLOGO CROSS-VALIDATION COMPLETE")
    print(f"Mean RMSE: {results['mean_rmse']:.4f}")
    print(f"Std RMSE: {results['std_rmse']:.4f}")
    print(f"Min RMSE: {results['min_rmse']:.4f}")
    print(f"Max RMSE: {results['max_rmse']:.4f}")
    
    return results


def create_selection_table(pipeline, X, y, groups):
    """
    Create per-scenario selection table as required by brief.
    Returns DataFrame with: Demand ID, ML Selected Plant, Optimal Plant, Error
    """
    print("\nCreating per-scenario selection table")
    
    # Get predictions
    y_pred = pipeline.predict(X)
    
    # Extract plant IDs from original merged data (hacky but works)
    # We need to track which plant each row corresponds to
    # This assumes X still has access to original indices
    
    results = []
    unique_demands = np.unique(groups)
    
    for demand_id in unique_demands:
        mask = groups == demand_id
        true_costs = y[mask]
        pred_costs = y_pred[mask]
        
        # Find optimal plant (lowest true cost)
        optimal_idx = np.argmin(true_costs)
        optimal_cost = true_costs[optimal_idx]
        
        # Find ML selected plant (lowest predicted cost)
        selected_idx = np.argmin(pred_costs)
        selected_cost = true_costs[selected_idx]
        
        # Calculate error
        error = optimal_cost - selected_cost
        
        results.append({
            'Demand_ID': demand_id,
            'Optimal_Cost': optimal_cost,
            'Selected_Cost': selected_cost,
            'Error': error
        })
    
    selection_df = pd.DataFrame(results)
    print(f"Created selection table for {len(selection_df)} demands")
    
    return selection_df