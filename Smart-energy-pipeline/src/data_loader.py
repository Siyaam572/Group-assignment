# src/data_loader.py

import pandas as pd

def load_data(config):
    """
    Loads demand, plants, and costs CSVs and merges them.
    Returns a dict with merged_df and raw dfs.
    """

    demand_path = config["data"]["demand_path"]
    plants_path = config["data"]["plants_path"]
    costs_path = config["data"]["costs_path"]

    demand = pd.read_csv(demand_path)
    plants = pd.read_csv(plants_path)
    costs = pd.read_csv(costs_path)

    # Merge costs with demand, then plants
    merged = (
        costs
        .merge(demand, on="Demand ID", how="inner")
        .merge(plants, on="Plant ID", how="inner")
    )

    return {
        "demand_df": demand,
        "plants_df": plants,
        "costs_df": costs,
        "merged_df": merged
    }
