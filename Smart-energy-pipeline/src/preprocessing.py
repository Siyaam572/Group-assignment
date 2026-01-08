"""
Data preprocessing module
"""
import pandas as pd
import numpy as np


def preprocess_data(data, config):
    """
    Preprocess data for model training.
    
    Args:
        data: Dictionary with 'demand', 'plants', 'costs', 'merged_df' keys
        config: Configuration dictionary
    
    Returns:
        X: Feature matrix (pandas DataFrame)
        y: Target values (pandas Series)
        groups: Demand IDs (pandas Series)
        plant_ids: Plant IDs (pandas Series)
        df_full: Full merged DataFrame
    """
    print("\nStarting data preprocessing...")
    
    # Extract dataframes - data_loader already provides these
    demand = data['demand_df'].copy()
    plants = data['plants_df'].copy()
    costs = data['costs_df'].copy()
    
    # Handle missing values
    print("Handling missing values...")
    demand_missing = 0
    for col in demand.columns:
        if demand[col].dtype in ['float64', 'int64'] and demand[col].isnull().any():
            median_val = demand[col].median()
            demand[col] = demand[col].fillna(median_val)
            demand_missing += demand[col].isnull().sum()
    print(f"Filled {demand_missing} missing values in demand features")
    
    cost_missing = costs['Cost_USD_per_MWh'].isnull().sum()
    median_cost = costs['Cost_USD_per_MWh'].median()
    costs['Cost_USD_per_MWh'] = costs['Cost_USD_per_MWh'].fillna(median_cost)
    print(f"Filled {cost_missing} missing values in costs")
    
    # Fix data quality issues
    print("Fixing data quality issues...")
    demand['DF_region'] = demand['DF_region'].replace('NA', 'NORAM')
    plants['Region'] = plants['Region'].replace('NA', 'NORAM')
    
    # Filter non-competitive plants
    print("Filtering non-competitive plants...")
    top_plants_per_demand = costs.sort_values(['Demand ID', 'Cost_USD_per_MWh']).groupby('Demand ID').head(10)
    competitive_plants = top_plants_per_demand['Plant ID'].unique()
    
    original_plant_count = plants['Plant ID'].nunique()
    plants = plants[plants['Plant ID'].isin(competitive_plants)]
    costs = costs[costs['Plant ID'].isin(competitive_plants)]
    remaining_plants = plants['Plant ID'].nunique()
    
    print(f"Removed {original_plant_count - remaining_plants} plants, {remaining_plants} remaining")
    
    # Merge datasets
    print("Merging datasets...")
    df = costs.merge(demand, on='Demand ID', how='inner')
    df = df.merge(plants, on='Plant ID', how='inner')
    print(f"Merged dataset shape: {df.shape}")
    
    # Separate features and target
    categorical_cols = ['DF_region', 'DF_daytype', 'Plant Type', 'Region']
    
    # One-hot encode categorical variables
    print(f"One-hot encoding {len(categorical_cols)} categorical columns")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Get final feature columns
    encoded_feature_cols = [c for c in df_encoded.columns if c.startswith('DF') or c.startswith('PF')]
    
    # Extract data - KEEP AS DATAFRAME/SERIES (not numpy arrays!)
    X = df_encoded[encoded_feature_cols]
    y = df_encoded['Cost_USD_per_MWh']
    groups = df_encoded['Demand ID']
    plant_ids = df_encoded['Plant ID']
    
    print("\nPreprocessing complete")
    print(f"Features: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Unique demands: {groups.nunique()}, Unique plants: {plant_ids.nunique()}")
    
    return X, y, groups, plant_ids, df_encoded
