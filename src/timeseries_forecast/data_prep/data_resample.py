import pandas as pd
from typing import Optional

def resample_dataframe_with_aggregations(df, frequency, agg_config=None):
    """
    Resample a dataframe with different aggregation functions for different column types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing OHLCV data
    frequency : str
        Resampling frequency (e.g., '24h', '1D', '4H')
    agg_config : dict, optional
        Configuration mapping column names or patterns to aggregation functions.
        If None, uses automatic detection based on column names.
        Example: {'Close': 'last', 'Volume': 'sum', 'High': 'max', 'Low': 'min'}
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe with all columns combined
    """
    
    if agg_config is None:
        # Default configuration based on column name patterns
        agg_config = {}
        
    # Make a copy to avoid modifying the original
    df_resample = df.copy()
    resampled_data = {}
    
    for col in df_resample.columns:
        # Determine aggregation function
        agg_func = None
        
        # Check if column is explicitly in agg_config
        if col in agg_config:
            agg_func = agg_config[col]
        else:
            # Auto-detect based on column name patterns
            col_lower = col.lower()
            if 'close' in col_lower:
                agg_func = 'last'
            elif 'open' in col_lower:
                agg_func = 'first'
            elif 'high' in col_lower:
                agg_func = 'max'
            elif 'low' in col_lower:
                agg_func = 'min'
            elif (
                'volume' in col_lower
                or 'buy' in col_lower
                or 'number of trades' in col_lower
                or 'number' in col_lower
            ):
                agg_func = 'sum'
            else:
                # Default to last for unknown columns
                agg_func = 'last'
        
        # Apply aggregation
        resampled_data[col] = getattr(df_resample[col].resample(frequency), agg_func)()
    
    # Combine all resampled series into a dataframe
    result_df = pd.concat(resampled_data.values(), axis=1, keys=resampled_data.keys())
    
    return result_df

# Example usage:
# resampled_df = resample_dataframe_with_aggregations(df, '24h')
print("OK: Function 'resample_dataframe_with_aggregations' defined")