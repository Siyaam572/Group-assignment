"""
Model training module using pipeline architecture
Pipeline: StandardScaler -> VarianceThreshold -> Regressor
"""
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logger = logging.getLogger(__name__)


def train_model(X, y, groups, plant_ids, config):
    """
    Train model using sklearn Pipeline architecture.
    
    Pipeline steps:
    1. StandardScaler - standardize features
    2. VarianceThreshold - remove low-variance features
    3. Regressor - RandomForest or GradientBoosting
    
    Args:
        X: Feature matrix
        y: Target values (costs)
        groups: Demand IDs for grouping (not used in training but kept for consistency)
        plant_ids: Plant IDs (not used but kept for consistency)
        config: Configuration dictionary
    
    Returns:
        model: Trained sklearn Pipeline object
        pipeline: Same as model (for compatibility)
    """
    logger.info("Training model with pipeline architecture...")
    
    model_type = config['model']['type']
    
    # Select regressor based on config
    if model_type == 'GradientBoosting':
        regressor = GradientBoostingRegressor(random_state=42)
        logger.info("Using GradientBoostingRegressor")
    elif model_type == 'RandomForest':
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        logger.info("Using RandomForestRegressor")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("regressor", regressor)
    ])
    
    # Train pipeline
    logger.info("Fitting pipeline on full dataset...")
    pipeline.fit(X, y)
    
    logger.info(f"Model training complete: {model_type}")
    
    return pipeline, pipeline
