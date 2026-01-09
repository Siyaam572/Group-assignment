import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from src.model_trainer import create_pipeline


def get_param_grid(model_type):
    # Parameter grid for hyperparameter search
    # Need to prefix with 'model__' for Pipeline
    if model_type == 'gradient_boosting':
        param_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 4, 5]
        }
    elif model_type == 'random_forest':
        param_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return param_grid


def tune_hyperparameters(model_type, preprocessor, X, y, groups, config):
    """
    Perform hyperparameter tuning using GridSearchCV with LOGO CV.
    If tuning is disabled, returns pipeline with pre-tuned params.
    """
    tuning_enabled = config.get('tuning', {}).get('enabled', False)
    
    if not tuning_enabled:
        print("\nHyperparameter tuning disabled in config")
        print("Using pre-optimized parameters from individual assessments")
        pipeline = create_pipeline(model_type, preprocessor, config)
        return pipeline, None
    
    print("\nStarting hyperparameter tuning")
    
    base_pipeline = create_pipeline(model_type, preprocessor, config)
    param_grid = get_param_grid(model_type)
    
    print(f"Parameter grid: {param_grid}")
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total combinations: {total_combinations}")
    
    logo = LeaveOneGroupOut()
    n_jobs = config.get('cross_validation', {}).get('n_jobs', -1)
    
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=logo.split(X, y, groups),
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=2
    )
    
    print("\nRunning GridSearchCV (this may take a while)")
    grid_search.fit(X, y, groups=groups)
    
    print("\nTuning complete")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_:.4f}")
    
    tuning_results = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, tuning_results