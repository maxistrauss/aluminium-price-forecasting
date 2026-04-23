from typing import List, Tuple
import pandas as pd
from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)

ALLOWED_AGG_FUNCS = ["mean", "std", "min", "max", "median", "skew", "kurt", "sum", "quantile"]
SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}


def add_lags(
        df: pd.DataFrame,
        lags: List[int],
        column: str,
        ts_id: str = "type",
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    col_dict = {
        f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
    }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features


def add_rolling_features(
        df: pd.DataFrame,
        rolls: List[int],
        column: str,
        ts_id: str = "type",
        agg_funcs: List[str] = ["mean", "std"],
        n_shift: int = 1,
) -> Tuple[pd.DataFrame, List]:
    """Add rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        rolls (List[int]): Different windows over which the rolling aggregations to be done
        column (str): The column used for feature engineering
        ts_id (str): Unique id for a time series.
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].
        n_shift (int, optional): Number of time steps to shift before computing rolling statistics.
            Typically used to avoid data leakage. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    agg_funcs = [agg if type(agg) is tuple else (agg, agg) for agg in agg_funcs]

    rolling_df = pd.concat(
        [
            df.groupby(ts_id)[column]
            .shift(n_shift)
            .rolling(l)
            .agg({f"{column}_rolling_{l}_{agg_name}": agg_func for agg_name, agg_func in agg_funcs})
            for l in rolls
        ],
        axis=1,
    )

    df = df.assign(**rolling_df.to_dict("list"))
    added_features = rolling_df.columns.tolist()
    return df, added_features


def add_seasonal_rolling_features(
        df: pd.DataFrame,
        seasonal_periods: List[int],
        rolls: List[int],
        column: str,
        ts_id: str = "type",
        agg_funcs: List[str] = ["mean", "std"],
        n_shift: int = 1,
) -> Tuple[pd.DataFrame, List]:
    """Add seasonal rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        seasonal_periods (List[int]): List of seasonal periods over which the seasonal rolling operations should be done
        rolls (List[int]): List of seasonal rolling window over which the aggregation functions will be applied
        column (str): [description]
        ts_id (str): Unique id for a time series.
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].. Defaults to ["mean", "std"].
        n_shift (int, optional): The number of seasonal shifts to be applied before the seasonal rolling operation.
            Typically used to avoid data leakage. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert (
            len(set(agg_funcs) - set(ALLOWED_AGG_FUNCS)) == 0
    ), f"`agg_funcs` should be one of {ALLOWED_AGG_FUNCS}"
    agg_funcs = {agg: SEASONAL_ROLLING_MAP[agg] for agg in agg_funcs}
    added_features = []

    for sp in seasonal_periods:
        season_shifts = (n_shift // sp) + 1
        col_dict = {
            f"{column}_{sp}_seasonal_rolling_{l}_{name}": df.groupby(ts_id)[
                column
            ].transform(
                lambda x: agg(
                    x.shift(season_shifts * sp).values,
                    season_length=sp,
                    window_size=l,
                )
            )
            for (name, agg) in agg_funcs.items()
            for l in rolls
        }
        df = df.assign(**col_dict)
        added_features += list(col_dict.keys())
    return df, added_features


def add_ewma(
        df: pd.DataFrame,
        column: str,
        ts_id: str = "type",
        alphas: List[float] = [0.5],
        spans: List[float] = None,
        n_shift: int = 1,
) -> Tuple[pd.DataFrame, List]:
    """Create Exponentially Weighted Average for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        column (str): Name of the column to be lagged
        ts_id (str): Unique ID of a time series to be grouped by before applying the lags.
        alphas (List[float]): List of alphas (smoothing parameters) using which ewmas are be created
        spans (List[float]): List of spans using which ewmas are be created. When we refer to a 60 period EWMA, span is 60.
            alpha = 2/(1+span). If span is given, we ignore alpha.
        n_shift (int, optional): Number of time steps to shift before computing ewma.
            Typically used to avoid data leakage. Defaults to 1.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    use_spans = False
    if spans is not None:
        assert isinstance(
            spans, list
        ), "`spans` should be a list of all required period spans"
        use_spans = True
    if alphas is not None:
        assert isinstance(
            alphas, list
        ), "`alphas` should be a list of all required smoothing parameters"
    if spans is None and alphas is None:
        raise ValueError(
            "Either `alpha` or `spans` should be provided for the function to"
        )
    if spans is None and alphas is None:
        raise ValueError(
            "Either `alpha` or `spans` should be provided for the function to"
        )
    assert (
            column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    col_dict = {
        f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df.groupby(
            [ts_id]
        )[column]
        .shift(n_shift)
        .ewm(
            alpha=None if use_spans else param,
            span=param if use_spans else None,
            adjust=False,
        )
        .mean()
        for param in (spans if use_spans else alphas)
    }
    df = df.assign(**col_dict)
    return df, list(col_dict.keys())


if __name__ == "__main__":
    df = pd.read_csv("../../tests/test.csv")
    df1, added_features = add_rolling_features(df, [3, 5], "demand_ub")
    df.loc[df["mode"] == "test", "demand_ub"] = 0
    df2, added_features2 = add_rolling_features(df, [3, 5], "demand_ub")

    print(added_features)