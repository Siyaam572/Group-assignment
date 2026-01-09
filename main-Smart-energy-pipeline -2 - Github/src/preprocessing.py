import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def fix_na_values(demand, plants):
    """Replace 'NA' string with 'NORAM'"""
    print("Fixing NA values")
    
    if 'DF_region' in demand.columns:
        demand['DF_region'] = demand['DF_region'].replace('NA', 'NORAM')
    
    if 'Region' in plants.columns:
        plants['Region'] = plants['Region'].replace('NA', 'NORAM')
    
    return demand, plants


def handle_missing_values(demand, plants, costs):
    # Fill missing values with median
    
    
    demand_missing = demand.isna().sum().sum()
    costs_missing = costs.isna().sum().sum()
    
    if demand_missing > 0:
        print(f"Missing in demand: {demand_missing}")
    if costs_missing > 0:
        print(f"Missing in costs: {costs_missing}")
    
    for col in demand.columns:
        if demand[col].dtype == 'float64' and demand[col].isnull().any():
            median_val = demand[col].median()
            demand[col] = demand[col].fillna(median_val)
    
    if costs['Cost_USD_per_MWh'].isnull().any():
        median_cost = costs['Cost_USD_per_MWh'].median()
        costs['Cost_USD_per_MWh'] = costs['Cost_USD_per_MWh'].fillna(median_cost)
    
    return demand, plants, costs


def filter_plants(costs, plant_id_col, cost_col, top_k=52):
    # Keep only competitive plants
    print(f"\nAnalyzing plant performance")
    
    top_plants_per_demand = costs.groupby('Demand ID').apply(
        lambda x: x.nsmallest(10, cost_col)[plant_id_col].tolist()
    )
    
    all_top_plants = []
    for plants_list in top_plants_per_demand:
        all_top_plants.extend(plants_list)
    
    plant_counts = pd.Series(all_top_plants).value_counts()
    plants_to_keep = plant_counts.index.tolist()
    
    removed = 64 - len(plants_to_keep)
    print(f"Original plants: 64")
    print(f"Removing {removed} plants that never appeared in top 10")
    print(f"Keeping {len(plants_to_keep)} plants")
    
    return plants_to_keep


def merge_datasets(demand, plants, costs, demand_id_col, plant_id_col):
    print("\nMerging datasets")
    
    merged = costs.merge(demand, on=demand_id_col, how='inner')
    merged = merged.merge(plants, on=plant_id_col, how='inner')
    
    print(f"Merged shape: {merged.shape}")
    
    return merged


def create_preprocessor(numerical_cols, categorical_cols):
    
    
    transformers = [
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_cols)
    ]
    
    # Pass through numerical columns unchanged
    
    transformers.append(
        ('num', FunctionTransformer(), numerical_cols)
    )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor


def preprocess_data(demand, plants, costs, config):
    
    demand, plants = fix_na_values(demand, plants)
    demand, plants, costs = handle_missing_values(demand, plants, costs)
    
    demand_id_col = 'Demand ID'
    plant_id_col = 'Plant ID'
    cost_col = 'Cost_USD_per_MWh'
    
    top_k = config.get('preprocessing', {}).get('top_k_plants', 52)
    plants_to_keep = filter_plants(costs, plant_id_col, cost_col, top_k)
    
    plants = plants[plants[plant_id_col].isin(plants_to_keep)]
    costs = costs[costs[plant_id_col].isin(plants_to_keep)]
    
    print(f"Filtered to {len(plants)} plants")
    
    merged = merge_datasets(demand, plants, costs, demand_id_col, plant_id_col)
    
    # Identify feature columns
    demand_features = [c for c in merged.columns if c.startswith('DF') and 
                      c not in ['DF_region', 'DF_daytype']]
    plant_features = [c for c in merged.columns if c.startswith('PF')]
    
    categorical_cols = []
    for col in ['DF_region', 'DF_daytype', 'Plant Type', 'Region']:
        if col in merged.columns:
            categorical_cols.append(col)
    
    numerical_cols = demand_features + plant_features
    
    print(f"\nPreparing features")
    print(f"  Numerical: {len(numerical_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")
    
    # Build feature DataFrame 
    feature_cols = numerical_cols + categorical_cols
    X = merged[feature_cols].copy()
    y = merged[cost_col].values
    groups = merged[demand_id_col].values
    
    # Create the preprocessor 
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    print(f"\nPreprocessing complete")
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features (before encoding)")
    print(f"Unique demands: {len(np.unique(groups))}")
    
    return X, y, groups, preprocessor