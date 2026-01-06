# src/model_trainer.py

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def train_model(X, y, groups, config):
    """
    Trains a model using config settings.
    Returns:
        model (fitted)
        pipeline (same as model for now)
    """

    model_type = config["model"]["type"]

    if model_type == "GradientBoosting":
        params = config["model"]["gradient_boosting"]
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
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
        raise ValueError(f"Unknown model type in config: {model_type}")

    model.fit(X, y)

    # For now "pipeline" is just model
    return model, model
