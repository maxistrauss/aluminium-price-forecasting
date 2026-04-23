from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from timeseries_forecast.data_prep.constants import (
    RawBitcoinColumn,
    RawEthColumn,
    RawGoldPriceColumn,
    RawSP500Column,
)
from timeseries_forecast.data_prep.data_merge import merge_all_raw_data
from timeseries_forecast.data_prep.data_resample import resample_dataframe_with_aggregations
from timeseries_forecast.data_prep.pipeline import run_data_pipeline
from timeseries_forecast.data_prep.raw_data import (
    read_raw_bitcoin_data,
    read_raw_eth_data,
    read_raw_gold_data,
    read_raw_sp500_data,
)
from timeseries_forecast.feature.custom_features import add_log_ret_features
from timeseries_forecast.feature.engineer_features import (
    FeatureEngineeringConfig,
    LaggingFeatures,
    RollingFeatures,
)


@dataclass
class DartsPreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    covariate_cols: List[str]


def _load_and_merge_base_dataframe() -> pd.DataFrame:
    btc_df_cleaned = run_data_pipeline(
        read_raw_bitcoin_data,
        timestamp_column=RawBitcoinColumn.OPEN_TIME,
    )
    eth_df_cleaned = run_data_pipeline(
        read_raw_eth_data,
        timestamp_column=RawEthColumn.OPEN_TIME,
    )
    gold_df_cleaned = run_data_pipeline(
        read_raw_gold_data,
        timestamp_column=RawGoldPriceColumn.OPEN_TIME,
    )
    sp500_df_cleaned = run_data_pipeline(
        read_raw_sp500_data,
        timestamp_column=RawSP500Column.OPEN_TIME,
    )

    merged_df = merge_all_raw_data(
        btc_df_cleaned=btc_df_cleaned,
        eth_df_cleaned=eth_df_cleaned,
        gold_df_cleaned=gold_df_cleaned,
        sp500_df_cleaned=sp500_df_cleaned,
    )
    return merged_df.sort_index()


def _build_feature_engineering_config(numeric_feature_columns: List[str], lag_window_len: int) -> FeatureEngineeringConfig:
    lag_features = [
        LaggingFeatures(column=column, lags=list(range(1, lag_window_len + 1)))
        for column in numeric_feature_columns
    ]

    rolling_features = [
        RollingFeatures(
            column=column,
            rolls=[3, 6, 12, 24],
            agg_funcs=["mean", "std", "min", "max"],
        )
        for column in numeric_feature_columns
    ]

    return FeatureEngineeringConfig(features=lag_features + rolling_features)


def prepare_darts_dataset(
    frequency_resample: str = "24h",
    forecast_horizon: int = 1,
    lag_window_len: int = 24,
    target_col: str = "BTC_Close_log_ret",
    include_contemporaneous_eth_log_ret: bool = False,
) -> DartsPreparedData:
    """Prepare train/val/test DataFrames for Darts deep-learning models.

    The pipeline mirrors the notebook logic:
    - run source-specific cleaning pipelines
    - merge BTC/ETH/GOLD/SP500
    - resample to chosen timeframe
    - add log-return features
    - generate lag and rolling features
    - split by BTC_Mode into train/val/test

    Notes
    -----
    If `include_contemporaneous_eth_log_ret=True`, a leaky proxy covariate
    `ETH_Close_log_ret_leaky_t = ETH_Close_log_ret.shift(-forecast_horizon)`
    is created so the covariate history at t-1 can contain ETH information at t.
    This is intentionally leaky and should only be used for controlled experiments.
    """
    merged_df = _load_and_merge_base_dataframe()
    resampled_df = resample_dataframe_with_aggregations(merged_df, frequency_resample)

    feature_df = resampled_df.copy()
    feature_df["type"] = "BTC"
    feature_df["timestamp"] = feature_df.index

    feature_df, _ = add_log_ret_features(feature_df, "BTC_Close")
    feature_df, _ = add_log_ret_features(feature_df, "ETH_Close")

    numeric_feature_columns = [
        "BTC_Open",
        "BTC_High",
        "BTC_Low",
        "BTC_Close",
        "BTC_Volume",
        "BTC_Quote asset volume",
        "BTC_Number of trades",
        "BTC_Taker buy base asset volume",
        "BTC_Taker buy quote asset volume",
        "ETH_Open",
        "ETH_High",
        "ETH_Low",
        "ETH_Close",
        "ETH_Volume",
        "GOLD_Open",
        "GOLD_High",
        "GOLD_Low",
        "GOLD_Close",
        "GOLD_Volume",
        "SP500_Open",
        "SP500_High",
        "SP500_Low",
        "SP500_Close",
        "SP500_Volume",
        "ETH_Close_log_ret",
        "BTC_Close_log_ret",
    ]

    feature_engineer = _build_feature_engineering_config(
        numeric_feature_columns=numeric_feature_columns,
        lag_window_len=lag_window_len,
    )

    features_df, created_feature_names = feature_engineer.create_features(
        feature_df,
        forecast_horizon=forecast_horizon,
    )

    if include_contemporaneous_eth_log_ret:
        leaky_eth_col = "ETH_Close_log_ret_leaky_t"
        features_df[leaky_eth_col] = features_df["ETH_Close_log_ret"].shift(-forecast_horizon)
        eth_covariate_col = leaky_eth_col
        print("âš ï¸ Using leaky ETH covariate (contains future/contemporaneous information).")
    else:
        eth_covariate_col = "ETH_Close_log_ret"

    selected_columns = [
        "timestamp",
        "BTC_Mode",
        target_col,
        eth_covariate_col,
    ]

    if include_contemporaneous_eth_log_ret:
        selected_columns.append("ETH_Close_log_ret")

    features_df = features_df[selected_columns].dropna().sort_values("timestamp") # type : ignore
    print("testtest2")

    train_df = (
        features_df[features_df["BTC_Mode"] == "Train"]
        .drop(columns=["BTC_Mode"])
        .set_index("timestamp")
        .sort_index()
    )
    val_df = (
        features_df[features_df["BTC_Mode"] == "Val"]
        .drop(columns=["BTC_Mode"])
        .set_index("timestamp")
        .sort_index()
    )
    test_df = (
        features_df[features_df["BTC_Mode"] == "Test"]
        .drop(columns=["BTC_Mode"])
        .set_index("timestamp")
        .sort_index()
    )

    covariate_cols = [column for column in train_df.columns if column != target_col]

    return DartsPreparedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col,
        covariate_cols=covariate_cols,
    )

