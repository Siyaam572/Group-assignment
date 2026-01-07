"""
Hyperparameter tuning using GridSearchCV with Leave-One-Group-Out CV

"""
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


def selection_rmse(y_true, y_pred, demand_ids):
    """
    Calculate selection error RMSE using Equations 1 & 2 from assessment brief.
    
    For each demand scenario:
    - Error(d) = min{c(p,d) | p in P} - c(p_ML, d)
    - Where p_ML is the plant selected by ML (lowest predicted cost)
    
    Overall Score = sqrt(mean(Error^2))
    
    Args:
        y_true: Actual costs
        y_pred: Predicted costs
        demand_ids: Demand ID for each row (for grouping)
    
    Returns:
        RMSE of selection errors across all demand scenarios
    """
    errors = []
    for d in np.unique(demand_ids):
        mask = demand_ids == d
        true_costs = y_true[mask]
        pred_costs = y_pred[mask]

        # Optimal cost (best plant for this demand)
        best_true_cost = true_costs.min()
        
        # ML-selected plant (plant with lowest predicted cost)
        selected_idx = np.argmin(pred_costs)
        selected_true_cost = true_costs[selected_idx]

        # Error for this demand
        errors.append(best_true_cost - selected_true_cost)

    return np.sqrt(np.mean(np.array(errors) ** 2))


def make_logo_scorer(groups_all, X_all):
    """
    Create custom scorer for GridSearchCV that uses selection RMSE.
    
    This scorer is compatible with GridSearchCV's interface while still
    calculating the proper selection error metric from the assessment brief.
    
    Args:
        groups_all: All demand IDs
        X_all: Full feature matrix (for index lookup)
    
    Returns:
        Scorer function compatible with GridSearchCV
    """
    def scorer(estimator, X_val, y_val):
        # Locate which demands are in this validation fold
        idx = np.isin(X_all, X_val).all(axis=1)
        groups_val = groups_all[idx]

        # Get predictions
        y_pred = estimator.predict(X_val)
        
        # Calculate selection RMSE
        # Return negative because GridSearchCV maximizes scores
        return -selection_rmse(y_val, y_pred, groups_val)

    return scorer


def tune_hyperparameters(X, y, groups, plant_ids, config):
    """
    Perform hyperparameter tuning using GridSearchCV with LOGO CV.
    Uses Role 3's implementation with 80-demand subsampling for efficiency.
    
    Args:
        X: Feature matrix (numpy array)
        y: Target values (costs)
        groups: Demand IDs for grouping in LOGO CV
        plant_ids: Plant IDs (not used but kept for interface consistency)
        config: Configuration dictionary
    
    Returns:
        best_pipeline: Tuned model pipeline (trained on full data)
        results: Dictionary with best params, RMSE, and metadata
    """
    logger.info("Starting hyperparameter tuning ...")
    
    model_type = config['model']['type']
    
    
    MAX_LOGO_DEMANDS = 80
    unique_demands = np.unique(groups)
    
    if len(unique_demands) > MAX_LOGO_DEMANDS:
        logger.info(f"Subsampling to {MAX_LOGO_DEMANDS} demands for tuning efficiency")
        np.random.seed(42)
        selected_demands = np.random.choice(
            unique_demands,
            size=MAX_LOGO_DEMANDS,
            replace=False
        )
        mask = np.isin(groups, selected_demands)
        X_sample = X[mask]
        y_sample = y[mask]
        groups_sample = groups[mask]
    else:
        X_sample = X
        y_sample = y
        groups_sample = groups
    
    logger.info(f"Tuning on {len(np.unique(groups_sample))} demands, {len(X_sample)} total rows")
    
    # Set up model and parameter grid 
    if model_type == 'GradientBoosting':
        regressor = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [3, 5]
        }
        logger.info("Tuning GradientBoosting: 8 parameter combinations")
    else:  # RandomForest
        regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 20],
            "regressor__min_samples_split": [2, 5]
        }
        logger.info("Tuning RandomForest: 8 parameter combinations")
    
    # Create pipeline 
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("regressor", regressor)
    ])
    
    # Custom scorer using selection RMSE (Equations 1 & 2)
    custom_scorer = make_logo_scorer(groups_sample, X_sample)
    
    # Set up Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    
    # GridSearchCV
    logger.info("Running GridSearchCV with LOGO CV...")
    logger.info("This may take several minutes...")
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=logo.split(X_sample, y_sample, groups_sample),
        scoring=custom_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search on sample
    grid.fit(X_sample, y_sample)
    
    # Extract best results
    best_pipeline = grid.best_estimator_
    best_params = grid.best_params_
    sample_rmse = -grid.best_score_  # Convert back to positive RMSE
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Sample RMSE with best params: {sample_rmse:.4f}")
    
    # Train final model on FULL dataset with best parameters
    logger.info("Training final model on full dataset with best parameters...")
    best_pipeline.fit(X, y)
    
    # Prepare results
    results = {
        'best_params': best_params,
        'sample_rmse': sample_rmse,
        'n_demands_tuned': len(np.unique(groups_sample)),
        'total_demands': len(np.unique(groups)),
        'param_grid': param_grid
    }
    
    logger.info("Hyperparameter tuning complete!")
    
    return best_pipeline, results
