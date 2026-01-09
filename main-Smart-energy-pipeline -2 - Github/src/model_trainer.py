from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_model_params(model_type):
    # Get hyperparameters for the model
    if model_type == 'gradient_boosting':
        params = {
            'n_estimators': 150,
            'learning_rate': 0.2,
            'max_depth': 3,
            'random_state': 42
        }
        
    elif model_type == 'random_forest':
        params = {
            'n_estimators': 50,
            'max_depth': 3,
            'min_samples_split': 2,
            'random_state': 42
        }
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return params


def create_pipeline(model_type, preprocessor, config):
    
    print(f"\nCreating pipeline with {model_type}")
    
    params = get_model_params(model_type)
    print(f"Model parameters: {params}")
    
    if model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create sklearn Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("Pipeline created successfully")
    
    return pipeline


def train_pipeline(pipeline, X_train, y_train):
    # Train the pipeline (preprocessor + model together)
    print("\nTraining pipeline")
    pipeline.fit(X_train, y_train)
    print("Training complete")
    return pipeline