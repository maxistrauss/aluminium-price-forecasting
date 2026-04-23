import numpy as np
import pandas as pd
from typing import Tuple, List

def add_log_ret_features(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds log return features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name for which log returns should be calculated.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the modified DataFrame and a list of new feature names.
    """
    df = df.copy()
    feature_name = f"{column}_log_ret"
    df[feature_name] = np.log(df[column]) - np.log(df[column].shift(1))
    return df, [feature_name]