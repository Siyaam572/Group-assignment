"""
Model evaluation using Leave-One-Group-Out cross-validation
Uses selection RMSE metric from assessment brief (Equations 1 & 2)
"""
import logging
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

logger = logging.getLogger(__name__)


def calculate_selection_rmse(y_true, y_pred, demand_ids):
    """
    Calculate selection error RMSE using Equations 1 & 2 from assessment brief.
    
    For each demand scenario:
    - Find the optimal plant (minimum actual cost)
    - Find ML-selected plant (minimum predicted cost)
    - Error(d) = optimal_cost - ml_selected_cost
    
    Overall Score = sqrt(mean(Error^2))
    
    Args:
        y_true: Actual costs
        y_pred: Predicted costs  
        demand_ids: Demand ID for each row
    
    Returns:
        Dictionary with RMSE, mean error, std dev, and optimal count
    """
    errors = []
    optimal_count = 0
    
    for d in np.unique(demand_ids):
        mask = demand_ids == d
        true_costs = y_true[mask]
        pred_costs = y_pred[mask]

        # Optimal cost
        best_true_cost = true_costs.min()
        
        # ML-selected plant
        selected_idx = np.argmin(pred_costs)
        selected_true_cost = true_costs[selected_idx]

        # Error for this demand
        error = best_true_cost - selected_true_cost
        errors.append(error)
        
        # Count optimal selections (error = 0)
        if abs(error) < 0.01:
            optimal_count += 1

    errors = np.array(errors)
    
    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'optimal_selections': optimal_count,
        'total_demands': len(np.unique(demand_ids))
    }


def evaluate_model(model, X, y, groups, config, df_full=None, plant_ids=None):
    """
    Evaluate model using Leave-One-Group-Out cross-validation.
    Fast approach: trains once, reuses for all folds.
    
    Args:
        model: Trained model/pipeline to evaluate
        X: Feature matrix (numpy array)
        y: Target values (numpy array)
        groups: Demand IDs (numpy array)
        config: Configuration dictionary
        df_full: Full dataframe (optional, not used)
        plant_ids: Plant IDs (optional, not used)
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting Leave-One-Group-Out cross-validation...")
    logger.info("Using fast evaluation (reuse trained model)")
    
    logo = LeaveOneGroupOut()
    n_demands = len(np.unique(groups))
    
    logger.info(f"Evaluating on {n_demands} demand scenarios...")
    logger.info("This will take a few minutes...")
    
    # Collect predictions for all folds
    all_y_true = []
    all_y_pred = []
    all_groups = []
    
    fold_count = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        # Get predictions for this fold
        # X is a pandas DataFrame, so use .iloc for integer indexing
        y_pred = model.predict(X.iloc[test_idx])
        
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)
        all_groups.extend(groups[test_idx])
        
        fold_count += 1
        if fold_count % 100 == 0:
            logger.info(f"  Processed {fold_count}/{n_demands} demands...")
    
    logger.info(f"  Processed {fold_count}/{n_demands} demands...")
    
    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_groups = np.array(all_groups)
    
    # Calculate selection RMSE
    logger.info("Calculating selection RMSE...")
    results = calculate_selection_rmse(all_y_true, all_y_pred, all_groups)
    
    logger.info(f"\nCross-validation complete!")
    logger.info(f"Mean error: {results['mean_error']:.4f}")
    logger.info(f"Std dev of errors: {results['std_error']:.4f}")
    logger.info(f"RMSE: {results['rmse']:.4f}")
    logger.info(f"Number of optimal selections: {results['optimal_selections']}/{results['total_demands']}")
    
    return results
