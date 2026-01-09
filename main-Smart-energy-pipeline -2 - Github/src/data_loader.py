import os
import pandas as pd


DEMAND_FILE = "demand.csv"
PLANTS_FILE = "plants.csv"
COST_FILE = "generation_costs.csv"

def load_datasets(data_dir='data'):
    
    demand_path = os.path.join(data_dir, DEMAND_FILE)
    plants_path = os.path.join(data_dir, PLANTS_FILE)
    costs_path = os.path.join(data_dir, COST_FILE)
    
    demand_df = pd.read_csv(demand_path, keep_default_na=False, na_values=[''])
    plants_df = pd.read_csv(plants_path, keep_default_na=False, na_values=[''])
    costs_df = pd.read_csv(costs_path)
    
    print(f"Loaded demand: {demand_df.shape}")
    print(f"Loaded plants: {plants_df.shape}")
    print(f"Loaded costs: {costs_df.shape}")
    
    return demand_df, plants_df, costs_df

#Auto-detect column name by keywords
def detect_column(df, keywords):
    
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

#Find the important columns
def get_column_names(demand_df, plants_df, costs_df):

    demand_id = detect_column(demand_df, ['demand'])
    plant_id = detect_column(plants_df, ['plant'])
    cost_col = detect_column(costs_df, ['cost'])
    
    if not demand_id or not plant_id or not cost_col:
        raise ValueError("Could not find required columns")
    
    print(f"Found columns: {demand_id}, {plant_id}, {cost_col}")
    
    return demand_id, plant_id, cost_col