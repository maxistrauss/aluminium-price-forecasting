from abc import abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import List, Tuple

import pandas as pd

from timeseries_forecast.feature import autoregressive_features, temporal_features
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler

from timeseries_forecast.feature import autoregressive_features, temporal_features


@dataclass
class Feature:
    column: str = field(
        default=MISSING,
        metadata={"help": "Column name of the column of which the feature needs to be created"},
    )

    @abstractmethod
    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        pass


@dataclass
class LaggingFeatures(Feature):
    lags: List[int] = field(
        default_factory=list,
        metadata={"help": "List of lags to be created"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        lags = [lag + forecast_horizon - 1 for lag in self.lags]
        assert min(lags) >= forecast_horizon, f"Lags must be greater or equal to forecast horizon"
        return autoregressive_features.add_lags(df, lags, self.column)


@dataclass
class RollingFeatures(Feature):
    rolls: List[int] = field(
        default_factory=list,
        metadata={"help": "List of rolls to be created"},
    )
    agg_funcs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of aggregation functions to be applied"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        return autoregressive_features.add_rolling_features(df, self.rolls, self.column, agg_funcs=self.agg_funcs,
                                                            n_shift=forecast_horizon)


@dataclass
class SeasonalRollingFeatures(RollingFeatures):
    seasonal_periods: List[int] = field(
        default_factory=list,
        metadata={"help": "List of seasonal periods over which the seasonal rolling operations should be done"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        return autoregressive_features.add_seasonal_rolling_features(df, self.seasonal_periods, self.rolls, self.column,
                                                                     agg_funcs=self.agg_funcs,
                                                                     n_shift=forecast_horizon)


@dataclass
class EWMAFeatures(Feature):
    spans: List[float] = field(
        default_factory=lambda: [0.5],
        metadata={"help": "List of spans to be created"},
    )
    alphas: List[float] = field(
        default_factory=list,
        metadata={"help": "List of alphas to be created"},
    )
    agg_funcs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of aggregation functions to be applied"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        return autoregressive_features.add_ewma(df, self.column, spans=self.spans, alphas=self.alphas,
                                                n_shift=forecast_horizon)


@dataclass
class TemporalFeatures(Feature):
    frequency: str = field(
        default=MISSING,
        metadata={"help": "Frequency of the date column"},
    )
    add_elapsed: bool = field(
        default=True,
        metadata={"help": "Flag to add elapsed time"},
    )
    prefix: str = field(
        default=None,
        metadata={"help": "Prefix to be used for the newly created features"},
    )
    drop: bool = field(
        default=True,
        metadata={"help": "Flag to drop the date column"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        return temporal_features.add_temporal_features(df, self.column, self.frequency, self.add_elapsed, self.prefix,
                                                       self.drop)


@dataclass
class FourierFeatures(Feature):
    max_value: int = field(
        default=None,
        metadata={"help": "Max value of the date column"},
    )
    n_fourier_terms: int = field(
        default=1,
        metadata={"help": "Number of Fourier terms to be created"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        return temporal_features.add_fourier_features(df, self.column, self.max_value, self.n_fourier_terms)


@dataclass
class FeatureEngineeringConfig:
    features: List[Feature] = field(
        default_factory=list,
        metadata={"help": "List of features to be created"},
    )

    def create_features(self, df: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        assert forecast_horizon > 0, "Forecast horizon should be greater than 0"
        added_features = []
        for feature in self.features:
            df, added = feature.create_features(df, forecast_horizon)
            added_features.extend(added)
        return df, added_features
