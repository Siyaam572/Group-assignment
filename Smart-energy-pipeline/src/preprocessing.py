# src/preprocessing.py
import pandas as pd

def preprocess_data(data, config):
    """
    Clean and prepare data for modeling
    
    Performs data cleaning including handling missing values, 
    fixing data quality issues, removing non-competitive plants,
    and one-hot encoding categorical variables.
    
    Returns:
        X: Features (one-hot encoded, numeric)
        y: Target variable (Cost_USD_per_MWh)
        groups: Demand IDs for cross-validation
        plant_ids: Plant IDs for evaluation
    """
    
    # Get the raw dataframes
    demand = data['demand_df'].copy()
    plants = data['plants_df'].copy()
    costs = data['costs_df'].copy()
    
    print("Starting data preprocessing...")
    
    # Handle missing values in demand features
    print("Handling missing values...")
    df_cols = [c for c in demand.columns if c.startswith('DF')]
    missing_count = demand[df_cols].isnull().sum().sum()
    
    for col in df_cols:
        if demand[col].isnull().any():
            demand[col].fillna(demand[col].median(), inplace=True)
    
    if missing_count > 0:
        print(f"Filled {missing_count} missing values in demand features")
    
    # Handle missing values in costs
    cost_missing = costs['Cost_USD_per_MWh'].isnull().sum()
    if cost_missing > 0:
        costs['Cost_USD_per_MWh'].fillna(costs['Cost_USD_per_MWh'].median(), inplace=True)
        print(f"Filled {cost_missing} missing values in costs")
    
    # Fix "NA" string to "NORAM" 
    print("Fixing data quality issues...")
    if 'DF_region' in demand.columns:
        na_count = (demand['DF_region'] == 'NA').sum()
        if na_count > 0:
            demand['DF_region'] = demand['DF_region'].replace('NA', 'NORAM')
            print(f"Fixed {na_count} 'NA' values in demand regions")
    
    if 'Region' in plants.columns:
        na_count = (plants['Region'] == 'NA').sum()
        if na_count > 0:
            plants['Region'] = plants['Region'].replace('NA', 'NORAM')
            print(f"Fixed {na_count} 'NA' values in plant regions")
    
    # Remove non-competitive plants
    # These plants never appeared in top 10 for any demand scenario
    print("Filtering non-competitive plants...")
    non_competitive = ['P2', 'P4', 'P5', 'P6', 'P9', 'P13', 
                       'P39', 'P46', 'P47', 'P52', 'P58', 'P59']
    
    plants_before = len(plants)
    plants = plants[~plants['Plant ID'].isin(non_competitive)]
    costs = costs[~costs['Plant ID'].isin(non_competitive)]
    print(f"Removed {plants_before - len(plants)} plants, {len(plants)} remaining")
    
    # Merge the datasets
    print("Merging datasets...")
    merged = pd.merge(costs, demand, on='Demand ID', how='inner')
    merged = pd.merge(merged, plants, on='Plant ID', how='inner')
    print(f"Merged dataset shape: {merged.shape}")
    
    # Extract variables before dropping columns
    groups = merged['Demand ID'].copy()
    plant_ids = merged['Plant ID'].copy()
    y = merged['Cost_USD_per_MWh'].copy()
    
    # Create feature matrix
    X = merged.drop(columns=['Cost_USD_per_MWh', 'Demand ID', 'Plant ID'])
    
    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"One-hot encoding {len(categorical_cols)} categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Convert to numeric
    X = X.astype(float)
    y = y.astype(float)
    
    print("\nPreprocessing complete")
    print(f"Features: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Unique demands: {len(groups.unique())}, Unique plants: {len(plant_ids.unique())}")
    
    return X, y, groups, plant_ids
