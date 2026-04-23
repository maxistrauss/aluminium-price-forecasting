import pandas as pd
from timeseries_forecast.data_prep.constants import RawBitcoinColumn, RawEthColumn, RawGoldPriceColumn


def merge_all_raw_data(btc_df_cleaned, eth_df_cleaned, gold_df_cleaned, sp500_df_cleaned):
    
    # Add prefixes to distinguish columns from different sources (all columns, no timestamp exclusion)
    btc_df = btc_df_cleaned.rename(columns={col: f'BTC_{col}' for col in btc_df_cleaned.columns})
    eth_df = eth_df_cleaned.rename(columns={col: f'ETH_{col}' for col in eth_df_cleaned.columns})
    gold_df = gold_df_cleaned.rename(columns={col: f'GOLD_{col}' for col in gold_df_cleaned.columns})
    sp500_df = sp500_df_cleaned.rename(columns={col: f'SP500_{col}' for col in sp500_df_cleaned.columns})
    
    # Merge all dataframes on index (timestamp is already set as index)
    merged_df = btc_df.copy()
    merged_df = merged_df.merge(eth_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.merge(gold_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.merge(sp500_df, left_index=True, right_index=True, how='inner')
    
    # Sort by index (timestamp)
    merged_df = merged_df.sort_index()

    # Zeitraum einheitlich begrenzen (using index)
    #merged_df = merged_df[(merged_df.index < pd.to_datetime("2025-12-31 23:00:00.000000")) & 
    #                      (merged_df.index >= pd.to_datetime("2018-01-02"))]
    
    return merged_df
