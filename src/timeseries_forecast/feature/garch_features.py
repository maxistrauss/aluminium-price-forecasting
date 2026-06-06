"""
Volatility and cross-asset correlation features.

GARCH-like conditional volatility is approximated via EWMA variance —
this is the RiskMetrics (J.P. Morgan 1994) approach, which is the
one-decay-parameter special case of GARCH(1,1) and avoids the `arch`
package dependency while matching its practical output closely.

All features are *shifted by forecast_horizon* so that at prediction time t
the model only sees information up to t-1 (no look-ahead leakage).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# EWMA / GARCH-proxy volatility features
# ---------------------------------------------------------------------------

def add_garch_proxy_features(
    df: pd.DataFrame,
    column: str,
    spans: List[int] = (5, 10, 21),
    forecast_horizon: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add EWMA-based volatility features for *column*.

    Features added per span s:
      - {col}_ewma_vol_{s}d   : EWMA conditional std-dev (λ = 1 - 2/(s+1))
      - {col}_sq_ret_{s}d     : rolling mean of squared log-returns (realized var proxy)
      - {col}_vol_ratio_{s}d  : ratio of short EWMA vol to long EWMA vol (vol-of-vol signal)

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex, already sorted ascending.
    column : price column from which log-returns are derived.
    spans : EWMA half-life spans in trading days.
    forecast_horizon : shift applied to all features to prevent leakage.

    Returns
    -------
    (df_with_features, list_of_new_column_names)
    """
    df = df.copy()
    feature_names: List[str] = []
    spans = sorted(spans)

    log_ret_col = f"{column}_log_ret"
    if log_ret_col not in df.columns:
        df[log_ret_col] = np.log(df[column].clip(lower=1e-8)).diff()

    returns = df[log_ret_col]

    # Cache EWMA variance per span to avoid repeated computation
    ewma_vols: dict = {}
    for span in spans:
        ewma_var = returns.ewm(span=span, adjust=False).var()
        ewma_vol = np.sqrt(ewma_var.clip(lower=0))
        ewma_vols[span] = ewma_vol

        col_vol = f"{column}_ewma_vol_{span}d"
        df[col_vol] = ewma_vol.shift(forecast_horizon)
        feature_names.append(col_vol)

        col_sq = f"{column}_sq_ret_{span}d"
        df[col_sq] = (returns ** 2).rolling(span, min_periods=max(1, span // 2)).mean().shift(forecast_horizon)
        feature_names.append(col_sq)

    # Vol-ratio: short/long — captures whether current vol is abnormally high
    if len(spans) >= 2:
        short_span = spans[0]
        long_span = spans[-1]
        col_ratio = f"{column}_vol_ratio_{short_span}_{long_span}d"
        ratio = (ewma_vols[short_span] / ewma_vols[long_span].clip(lower=1e-8)).shift(forecast_horizon)
        df[col_ratio] = ratio
        feature_names.append(col_ratio)

    return df, feature_names


# ---------------------------------------------------------------------------
# Cross-metal rolling correlation features
# ---------------------------------------------------------------------------

def add_cross_correlation_features(
    df: pd.DataFrame,
    metal_price_columns: List[str],
    windows: List[int] = (20, 60),
    forecast_horizon: int = 1,
    add_alignment_index: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add pairwise rolling Pearson correlations between log-returns of all metals.

    Also adds a *trend_alignment* score: fraction of metals moving in the
    same direction as the target metal (first element of metal_price_columns).

    Parameters
    ----------
    df : DataFrame with DatetimeIndex.
    metal_price_columns : list of price columns, e.g. ['ALU_Close', 'COPPER_Close', ...]
    windows : rolling window lengths in rows (days for 1D data).
    forecast_horizon : shift to prevent leakage.
    add_alignment_index : whether to add the trend alignment feature.

    Returns
    -------
    (df_with_features, list_of_new_column_names)
    """
    df = df.copy()
    feature_names: List[str] = []

    # Compute log returns for each column (or re-use if already present)
    ret_cols: dict = {}
    for col in metal_price_columns:
        ret_col = f"{col}_log_ret"
        if ret_col not in df.columns:
            df[ret_col] = np.log(df[col].clip(lower=1e-8)).diff()
        ret_cols[col] = ret_col

    n_metals = len(metal_price_columns)

    for window in windows:
        min_p = max(2, window // 3)

        # Pairwise Pearson correlations
        for i in range(n_metals):
            for j in range(i + 1, n_metals):
                col_i = metal_price_columns[i]
                col_j = metal_price_columns[j]

                # Short readable names (prefix before '_')
                name_i = col_i.split("_")[0]
                name_j = col_j.split("_")[0]

                corr_col = f"corr_{name_i}_{name_j}_{window}d"
                corr = (
                    df[ret_cols[col_i]]
                    .rolling(window, min_periods=min_p)
                    .corr(df[ret_cols[col_j]])
                    .shift(forecast_horizon)
                )
                df[corr_col] = corr
                feature_names.append(corr_col)

        # Trend-alignment index: fraction of metals trending same direction as metal[0]
        if add_alignment_index and n_metals > 1:
            target_sign = np.sign(
                df[ret_cols[metal_price_columns[0]]]
                .rolling(window, min_periods=min_p)
                .mean()
            )
            others_sign = pd.concat(
                [
                    np.sign(
                        df[ret_cols[c]]
                        .rolling(window, min_periods=min_p)
                        .mean()
                    )
                    for c in metal_price_columns[1:]
                ],
                axis=1,
            )
            alignment = (others_sign.eq(target_sign, axis=0)).mean(axis=1)
            align_col = f"trend_alignment_{window}d"
            df[align_col] = alignment.shift(forecast_horizon)
            feature_names.append(align_col)

    return df, feature_names
