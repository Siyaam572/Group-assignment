import os
import yaml
import json
import shutil
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from src.data_loader import load_datasets
from src.preprocessing import preprocess_data
from src.model_trainer import create_pipeline, train_pipeline
from src.evaluator import grouped_train_test_split, evaluate_on_test_set, logo_cross_validation, create_selection_table


def save_artifacts(pipeline, results, config, selection_table):
    # Save all artifacts as required by brief
    artifacts_dir = config.get('output', {}).get('artifacts_dir', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"\nSaving artifacts to {artifacts_dir}/")
    
    # Save the complete pipeline 
    model_path = os.path.join(artifacts_dir, 'best_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline: {model_path}")
    
    # Save the preprocessor separately 
    preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
    joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_path)
    print(f"Saved preprocessor: {preprocessor_path}")
    
    # Save performance summary
    summary_path = os.path.join(artifacts_dir, 'performance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    print(f"Saved summary: {summary_path}")
    
    # Save per-scenario selection table 
    if selection_table is not None:
        selection_path = os.path.join(artifacts_dir, 'selection_results.csv')
        selection_table.to_csv(selection_path, index=False)
        print(f"Saved selection table: {selection_path}")
    
    # Save CV results
    if 'cv_results' in results:
        cv_path = os.path.join(artifacts_dir, 'cv_results.csv')
        pd.DataFrame({
            'fold': range(len(results['cv_results']['fold_scores'])),
            'rmse': results['cv_results']['fold_scores']
        }).to_csv(cv_path, index=False)
        print(f"Saved CV results: {cv_path}")
    
    # Save the config that was used 
    config_path = os.path.join(artifacts_dir, 'config_used.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config: {config_path}")


def main():
    print("\nNEC Smart Energy Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Load data
    print("Step 1: Loading datasets")
    demand, plants, costs = load_datasets('data')
    
    # Step 2: Preprocess (returns DataFrame + preprocessor)
    print("\nStep 2: Preprocessing")
    X, y, groups, preprocessor = preprocess_data(demand, plants, costs, config)
    
    # Step 3: Create Pipeline (preprocessor + model)
    print("\nStep 3: Creating Pipeline")
    model_type = config.get('model', {}).get('type', 'gradient_boosting')
    pipeline = create_pipeline(model_type, preprocessor, config)
    
    # Step 4: Train/test evaluation
    print("\nStep 4: Grouped train/test evaluation")
    test_size = config.get('evaluation', {}).get('test_size', 20)
    random_state = config.get('evaluation', {}).get('random_state', 42)
    
    X_train, X_test, y_train, y_test, groups_train, groups_test = grouped_train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state
    )
    
    # Create a fresh pipeline for train/test
    train_test_pipeline = create_pipeline(model_type, preprocessor, config)
    train_test_results = evaluate_on_test_set(
        train_test_pipeline, X_train, X_test, y_train, y_test, groups_test
    )
    
    # Step 5: LOGO cross-validation
    print("\nStep 5: LOGO cross-validation")
    cv_pipeline = create_pipeline(model_type, preprocessor, config)
    cv_results = logo_cross_validation(cv_pipeline, X, y, groups, n_jobs=-1)
    
    # Step 6: Train final pipeline on all data
    print("\nStep 6: Training final pipeline on all data")
    final_pipeline = create_pipeline(model_type, preprocessor, config)
    final_pipeline = train_pipeline(final_pipeline, X, y)
    
    # Step 7: Create per-scenario selection table
    print("\nStep 7: Creating selection table")
    selection_table = create_selection_table(final_pipeline, X, y, groups)
    
    # Step 8: Save everything
    print("\nStep 8: Saving artifacts")
    
    results = {
        'summary': {
            'model_type': model_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_test': {
                'r2_score': float(train_test_results['r2_score']),
                'mse': float(train_test_results['mse']),
                'custom_rmse': float(train_test_results['custom_rmse']),
                'mean_error': float(train_test_results['mean_error']),
                'num_test_demands': int(train_test_results['num_test_demands']),
                'num_optimal_selections': int(train_test_results['num_optimal_selections'])
            },
            'logo_cv': {
                'mean_rmse': float(cv_results['mean_rmse']),
                'std_rmse': float(cv_results['std_rmse']),
                'min_rmse': float(cv_results['min_rmse']),
                'max_rmse': float(cv_results['max_rmse'])
            }
        },
        'cv_results': cv_results
    }
    
    save_artifacts(final_pipeline, results, config, selection_table)
    
    print("\nPipeline complete!")
    print(f"Model: {model_type}")
    print(f"Train/Test RMSE: {train_test_results['custom_rmse']:.2f} USD/MWh")
    print(f"LOGO CV RMSE: {cv_results['mean_rmse']:.2f} +/- {cv_results['std_rmse']:.2f} USD/MWh")
    print(f"Results saved to artifacts/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()