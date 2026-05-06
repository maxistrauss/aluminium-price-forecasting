import pandas as pd
from timeseries_forecast.data_prep.constants import MERGE_DATASET_CONFIG


def merge_all_raw_data(**datasets):
    """
    Mergt mehrere Datensätze dynamisch basierend auf MERGE_DATASET_CONFIG in constants.py.
    
    Args:
        **datasets: Keyword-Argumente mit Datensätzen als DataFrames.
                   Die Namen müssen mit MERGE_DATASET_CONFIG übereinstimmen.
    
    Returns:
        pd.DataFrame: Zusammengefügter DataFrame mit Präfixen aus MERGE_DATASET_CONFIG.
    
    Beispiel:
        merged_df = merge_all_raw_data(
            btc_df_cleaned=btc_df,
            eth_df_cleaned=eth_df,
            gold_df_cleaned=gold_df,
            sp500_df_cleaned=sp500_df,
        )
    """
    merged_df = None
    
    # Iteriere über die in MERGE_DATASET_CONFIG konfigurierten Datensätze
    for prefix, param_name in MERGE_DATASET_CONFIG:
        if param_name not in datasets:
            raise ValueError(
                f"Datensatz '{param_name}' nicht übergeben. "
                f"Erforderliche Parameter: {[p[1] for p in MERGE_DATASET_CONFIG]}"
            )
        
        # Hole DataFrame und füge Präfix hinzu
        df = datasets[param_name]
        df_prefixed = df.rename(columns={col: f'{prefix}_{col}' for col in df.columns})
        
        # Merge mit vorherigen DataFrames
        if merged_df is None:
            merged_df = df_prefixed.copy()
        else:
            merged_df = merged_df.merge(df_prefixed, left_index=True, right_index=True, how='inner')
    
    # Sortiere nach Index (Timestamp)
    merged_df = merged_df.sort_index()
    
    return merged_df

