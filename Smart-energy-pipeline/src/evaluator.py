# src/evaluator.py
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

def evaluate_model(model, X, y, groups, config, df_full=None, plant_ids=None):
    """
    Evaluate model using Leave-One-Group-Out cross-validation
    Tests the trained model on each demand scenario
    """
    
    print("Starting Leave-One-Group-Out cross-validation...")
    print("This will take a few minutes...")
    
    # Convert to numpy
    if hasattr(X, 'values'):
        X_arr = X.values
    else:
        X_arr = X
        
    if hasattr(y, 'values'):
        y_arr = y.values
    else:
        y_arr = y
        
    if hasattr(groups, 'values'):
        groups_arr = groups.values
    else:
        groups_arr = np.array(groups)
    
    # Setup LOGO CV
    logo = LeaveOneGroupOut()
    errors = []
    
    # Count unique demands
    unique_demands = np.unique(groups_arr)
    total_demands = len(unique_demands)
    
    print(f"Evaluating on {total_demands} demand scenarios...")
    
    # For each demand scenario
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X_arr, y_arr, groups_arr), 1):
        
        if fold_num % 100 == 0:
            print(f"  Processed {fold_num}/{total_demands} demands...")
        
        # Get predictions for this demand's plants
        y_pred_test = model.predict(X_arr[test_idx])
        y_true_test = y_arr[test_idx]
        
        # Find optimal (actual minimum cost)
        optimal_cost = y_true_test.min()
        
        # Find ML-selected plant (minimum predicted cost)
        ml_selected_idx = np.argmin(y_pred_test)
        ml_selected_cost = y_true_test[ml_selected_idx]
        
        # Calculate error for this demand
        error = optimal_cost - ml_selected_cost
        errors.append(error)
    
    # Calculate RMSE
    errors = np.array(errors)
    cv_rmse = np.sqrt(np.mean(errors ** 2))
    
    print(f"\nCross-validation complete!")
    print(f"Mean error: {np.mean(errors):.4f}")
    print(f"Std dev of errors: {np.std(errors):.4f}")
    print(f"RMSE: {cv_rmse:.4f}")
    print(f"Number of optimal selections: {(errors == 0).sum()}/{len(errors)}")
    
    return {
        'cv_selection_rmse_mean': float(cv_rmse),
        'cv_selection_rmse_std': float(np.std(errors)),
        'mean_error': float(np.mean(errors)),
        'cv_folds': int(len(errors)),
        'optimal_selections': int((errors == 0).sum())
    }
