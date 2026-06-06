"""
Market regime detection via KMeans on rolling volatility + trend features.

Design principles:
  - Fitted ONLY on training data → no test leakage.
  - Rolling features shifted by forecast_horizon → no within-sample leakage.
  - Regimes are integer labels {0, 1, 2}; caller can map them to names
    after inspecting cluster centroids.
  - Stateless after fit: the scaler + KMeans are stored on the object so
    the same model can label new data consistently.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Unsupervised market regime detector.

    Features used for clustering
    ----------------------------
    vol   : EWMA standard deviation of log-returns (span=vol_window)
    trend : rolling mean of log-returns (window=trend_window)
    skew  : rolling skewness of log-returns (captures fat-tail periods)

    Usage
    -----
    >>> detector = RegimeDetector(n_regimes=3)
    >>> detector.fit(train_df, price_col="ALU_Close")
    >>> df_with_regime = detector.transform(full_df, price_col="ALU_Close")
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_window: int = 21,
        trend_window: int = 10,
        skew_window: int = 21,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.skew_window = skew_window
        self.random_state = random_state

        self._kmeans: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._fill_regime: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Build (vol, trend, skew) feature matrix from *df*."""
        log_ret = np.log(df[price_col].clip(lower=1e-8)).diff()

        vol = log_ret.ewm(span=self.vol_window, adjust=False).std()
        trend = log_ret.rolling(self.trend_window, min_periods=2).mean()
        skew = log_ret.rolling(self.skew_window, min_periods=5).skew()

        return pd.DataFrame({"vol": vol, "trend": trend, "skew": skew}, index=df.index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, price_col: str) -> "RegimeDetector":
        """
        Fit KMeans on the *training* data only.

        NaN rows (window warm-up) are dropped before fitting.

        Parameters
        ----------
        train_df  : training slice of the dataset (DatetimeIndex, sorted).
        price_col : name of the price column (e.g. 'ALU_Close').
        """
        feat = self._build_features(train_df, price_col).dropna()

        if len(feat) < self.n_regimes * 10:
            raise ValueError(
                f"Too few non-NaN training rows ({len(feat)}) for {self.n_regimes} regimes. "
                "Increase initial_train_size or reduce vol_window."
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(feat.values)

        self._kmeans = KMeans(
            n_clusters=self.n_regimes,
            n_init=20,
            random_state=self.random_state,
        )
        self._kmeans.fit(X_scaled)

        # Most common regime used to fill NaN warm-up rows
        labels = self._kmeans.labels_
        self._fill_regime = int(np.bincount(labels).argmax())

        return self

    def transform(
        self,
        df: pd.DataFrame,
        price_col: str,
        forecast_horizon: int = 1,
        regime_col_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Append a regime integer column to *df*.

        The regime at time t is computed from features up to t-1
        (shifted by forecast_horizon) so it is valid as a predictor.

        Parameters
        ----------
        df : full (train+val+test) DataFrame.
        price_col : same column used in fit().
        forecast_horizon : shift to prevent leakage.
        regime_col_name : override the auto-generated column name.

        Returns
        -------
        (df_with_regime_col, regime_col_name)
        """
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Call fit() before transform().")

        col_name = regime_col_name or f"{price_col}_regime"

        feat = self._build_features(df, price_col)
        feat_shifted = feat.shift(forecast_horizon)

        valid_mask = ~feat_shifted.isna().any(axis=1)
        regimes = pd.Series(self._fill_regime, index=df.index, dtype=int, name=col_name)

        if valid_mask.sum() > 0:
            X_scaled = self._scaler.transform(feat_shifted.loc[valid_mask].values)
            labels = self._kmeans.predict(X_scaled)
            regimes.loc[valid_mask] = labels

        df = df.copy()
        df[col_name] = regimes

        return df, col_name

    def regime_summary(self, train_df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Return a DataFrame describing each regime's average vol, trend, skew.
        Useful for naming regimes after fitting.
        """
        if self._kmeans is None:
            raise RuntimeError("Call fit() first.")

        feat = self._build_features(train_df, price_col).dropna()
        X_scaled = self._scaler.transform(feat.values)
        labels = self._kmeans.predict(X_scaled)
        feat["regime"] = labels

        summary = feat.groupby("regime")[["vol", "trend", "skew"]].mean()
        summary["n_obs"] = feat.groupby("regime").size()
        return summary
