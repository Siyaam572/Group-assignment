"""
Smart Energy Grid Pipeline
Main entry point - runs the entire ML pipeline

Usage:
    python main.py
"""

import yaml
import logging
from pathlib import Path

# Import modules (others will create these)
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_trainer import train_model
from src.evaluator import evaluate_model
from src.tuner import tune_hyperparameters

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main pipeline execution"""
    
    print("="*60)
    print("SMART ENERGY GRID OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Step 1: Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    setup_logging(config['logging']['level'])
    logging.info("Configuration loaded successfully")
    
    # Step 2: Load data
    print("\n[2/6] Loading data...")
    data = load_data(config)
    logging.info(f"Data loaded: {data['merged_df'].shape[0]} rows")
    
    # Step 3: Preprocess data
    print("\n[3/6] Preprocessing data...")
    X, y, groups = preprocess_data(data, config)
    logging.info(f"Features prepared: {X.shape[1]} features, {len(groups.unique())} demand scenarios")
    
    # Step 4: Train model
    print("\n[4/6] Training model...")
    model, pipeline = train_model(X, y, groups, config)
    logging.info(f"Model trained: {config['model']['type']}")
    
    # Step 5: Evaluate model
    print("\n[5/6] Evaluating model...")
    results = evaluate_model(model, X, y, groups, config)
    logging.info(f"Cross-validation RMSE: {results['cv_rmse']:.4f}")
    
    # Step 6: Hyperparameter tuning (optional)
    if config['tuning']['enabled']:
        print("\n[6/6] Tuning hyperparameters...")
        best_model, tuning_results = tune_hyperparameters(X, y, groups, config)
        logging.info(f"Best RMSE after tuning: {tuning_results['best_rmse']:.4f}")
    else:
        print("\n[6/6] Skipping hyperparameter tuning (disabled in config)")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nModel: {config['model']['type']}")
    print(f"Final RMSE: {results['cv_rmse']:.4f}")
    print(f"\nOutputs saved to:")
    print(f"  - Models: {config['output']['models_dir']}")
    print(f"  - Results: {config['output']['results_dir']}")
    print(f"  - Figures: {config['output']['figures_dir']}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()