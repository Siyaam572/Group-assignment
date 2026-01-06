# src/preprocessing.py

def preprocess_data(data, config):
    """
    Takes merged_df and returns:
    X = features
    y = target
    groups = Demand IDs (for Leave-One-Group-Out)
    """

    df = data["merged_df"].copy()

    target_col = "Cost_USD_per_MWh"
    group_col = "Demand ID"

    # Features: DF* + PF* (simple and consistent)
    df_cols = [c for c in df.columns if c.startswith("DF")]
    pf_cols = [c for c in df.columns if c.startswith("PF")]
    feature_cols = df_cols + pf_cols

    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    return X, y, groups
