# src/model_trainer.py

# src/model_trainer.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def train_model(X, y, groups, plant_ids, config):
    """
    Train the selected model from config.yaml.

    Inputs:
      X: DataFrame of features
      y: Series of target (Cost_USD_per_MWh)
      groups: Series of Demand IDs (not used here, but kept for pipeline consistency)
      config: dict from config.yaml

    Returns:
      model: fitted sklearn model
      pipeline: same as model (placeholder for future sklearn Pipeline use)
    """

    model_type = config["model"]["type"]

    if model_type == "GradientBoosting":
        params = config["model"]["gradient_boosting"]

        # NOTE: GradientBoostingRegressor uses max_depth inside its base estimator via "max_depth"
        # via parameter "max_depth" in sklearn works in newer versions through "max_depth" in params
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )

    elif model_type == "RandomForest":
        params = config["model"]["random_forest"]
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"],
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unsupported model type in config: {model_type}")

    # Fit
    model.fit(X, y)

    # For now pipeline == model (later you can replace with a sklearn Pipeline)
    pipeline = model

    return model, pipeline

