"""
Smart Energy Grid Pipeline
Run this file to execute the complete ML pipeline

Usage: python main.py
"""

import yaml
import logging
from pathlib import Path

# Import our custom modules
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_trainer import train_model
from src.evaluator import evaluate_model
from src.tuner import tune_hyperparameters


def load_config(config_path="config/config.yaml"):
    """Load settings from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(level="INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function - runs entire pipeline"""
    
    print("\nSMART ENERGY GRID OPTIMIZATION PIPELINE\n")
    
    # Load configuration
    print("[1/6] Loading configuration...")
    config = load_config()
    setup_logging(config['logging']['level'])
    logging.info("Configuration loaded successfully")
    
    # Load the three datasets
    print("\n[2/6] Loading data...")
    data = load_data(config)
    logging.info(f"Data loaded: {data['merged_df'].shape[0]} rows")
    
    # Clean and prepare the data
    print("\n[3/6] Preprocessing data...")
    X, y, groups, plant_ids, df_full = preprocess_data(data, config)
    logging.info(f"Features prepared: {X.shape[1]} features, {len(groups.unique())} demand scenarios")
    
    # Train the model
    print("\n[4/6] Training model...")
    model, pipeline = train_model(X, y, groups, plant_ids, config)
    logging.info(f"Model trained: {config['model']['type']}")
    
    # Evaluate using LOGO CV
    print("\n[5/6] Evaluating model...")
    results = evaluate_model(model, X, y, groups, config, df_full=df_full, plant_ids=plant_ids)
    logging.info(f"Cross-validation RMSE: {results['cv_rmse']:.4f}")
    
    # Run hyperparameter tuning if enabled
    if config['tuning']['enabled']:
        print("\n[6/6] Tuning hyperparameters...")
        best_model, tuning_results = tune_hyperparameters(X, y, groups, plant_ids, config)
        logging.info(f"Best RMSE after tuning: {tuning_results['best_rmse']:.4f}")
    else:
        print("\n[6/6] Skipping hyperparameter tuning (disabled in config)")
    
    # Print summary
    print("\nPIPELINE COMPLETED SUCCESSFULLY\n")
    print(f"Model: {config['model']['type']}")
    print(f"Final RMSE: {results['cv_selection_rmse_mean']:.4f}")
    print(f"\nOutputs saved to:")
    print(f"  - Models: {config['output']['models_dir']}")
    print(f"  - Results: {config['output']['results_dir']}")
    print(f"  - Figures: {config['output']['figures_dir']}")


if __name__ == "__main__":
    main()

