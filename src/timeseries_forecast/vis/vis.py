"""Reusable visualisation helpers for Bitcoin OHLCV DataFrames."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_DEFAULT_STYLE = "seaborn-v0_8-darkgrid"


def _apply_style() -> None:
    """Apply a consistent matplotlib style."""
    try:
        plt.style.use(_DEFAULT_STYLE)
    except OSError:
        pass  # fall back to default if style not available


# ---------------------------------------------------------------------------
# 1) Price / Volume / Range overview  (based on your pasted code)
# ---------------------------------------------------------------------------


def plot_price_volume_range(
    df: pd.DataFrame,
    title_prefix: str = "Bitcoin",
    figsize: tuple[int, int] = (15, 12),
) -> None:
    """Three-panel chart: Close price, Volume, and hourly Price Range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``Close``, ``Volume``, ``High``, ``Low``.
    title_prefix : str
        Prefix used in subplot titles (e.g. ``"Bitcoin"`` or ``"ETH"``).
    figsize : tuple[int, int]
        Figure size.
    """
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Close price
    axes[0].plot(df.index, df["Close"], linewidth=0.8)
    axes[0].set_title(f"{title_prefix} Close Price", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")
    axes[0].grid(True, alpha=0.3)

    # Volume
    axes[1].plot(df.index, df["Volume"], linewidth=0.8, color="orange")
    axes[1].set_title("Trading Volume", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Volume")
    axes[1].grid(True, alpha=0.3)

    # Price range
    price_range = df["High"] - df["Low"]
    axes[2].plot(df.index, price_range, linewidth=0.8, color="green")
    axes[2].set_title("Price Range (High − Low) per Hour", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Price Range (USD)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2) Candlestick-style OHLC chart (pure matplotlib, no mplfinance needed)
# ---------------------------------------------------------------------------


def plot_ohlc(
    df: pd.DataFrame,
    last_n: int = 168,
    title: str = "OHLC Chart",
    figsize: tuple[int, int] = (16, 6),
) -> None:
    """Simple OHLC bar chart for the last *last_n* candles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Open``, ``High``, ``Low``, ``Close`` columns.
    last_n : int
        Number of most recent candles to display (default 168 = 7 days).
    """
    _apply_style()
    subset = df.tail(last_n).copy()

    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(subset["Close"] >= subset["Open"], "green", "red")

    for i, (idx, row) in enumerate(subset.iterrows()):
        ax.plot([i, i], [row["Low"], row["High"]], color=colors[i], linewidth=0.8)
        ax.plot(
            [i, i],
            [row["Open"], row["Close"]],
            color=colors[i],
            linewidth=3,
        )

    # x-axis labels
    tick_step = max(1, len(subset) // 10)
    ax.set_xticks(range(0, len(subset), tick_step))
    ax.set_xticklabels(
        [subset.iloc[j].name.strftime("%Y-%m-%d %H:%M") for j in range(0, len(subset), tick_step)],
        rotation=45,
        ha="right",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3) Returns distribution
# ---------------------------------------------------------------------------


def plot_returns_distribution(
    df: pd.DataFrame,
    close_col: str = "Close",
    bins: int = 150,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Histogram + KDE of hourly log-returns with summary statistics."""
    _apply_style()
    log_ret = np.log(df[close_col] / df[close_col].shift(1)).dropna()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram
    axes[0].hist(log_ret, bins=bins, edgecolor="black", alpha=0.7)
    axes[0].axvline(log_ret.mean(), color="red", linestyle="--", label=f"Mean {log_ret.mean():.5f}")
    axes[0].set_title("Log-Returns Histogram", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Log Return")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(log_ret, vert=True)
    axes[1].set_title("Log-Returns Box Plot", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Log Return")

    # QQ-style: sorted returns vs normal quantiles
    sorted_ret = np.sort(log_ret.values)
    norm_quantiles = np.random.default_rng(42).normal(
        log_ret.mean(), log_ret.std(), size=len(sorted_ret)
    )
    norm_quantiles.sort()
    axes[2].scatter(norm_quantiles, sorted_ret, s=1, alpha=0.5)
    lim = max(abs(sorted_ret.min()), abs(sorted_ret.max()))
    axes[2].plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
    axes[2].set_title("QQ Plot (vs Normal)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")

    plt.tight_layout()
    plt.show()

    print(f"  Skewness: {log_ret.skew():.4f}")
    print(f"  Kurtosis: {log_ret.kurtosis():.4f}")


# ---------------------------------------------------------------------------
# 4) Rolling statistics (MA + volatility bands)
# ---------------------------------------------------------------------------


def plot_rolling_statistics(
    df: pd.DataFrame,
    close_col: str = "Close",
    windows: Optional[list[int]] = None,
    figsize: tuple[int, int] = (16, 10),
) -> None:
    """Plot close price with moving averages and rolling volatility.

    Parameters
    ----------
    windows : list[int]
        Rolling window sizes in hours (default ``[24, 168, 720]`` =
        1 day, 1 week, 30 days).
    """
    if windows is None:
        windows = [24, 168, 720]
    _apply_style()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Price + MAs
    axes[0].plot(df.index, df[close_col], linewidth=0.6, alpha=0.7, label="Close")
    colors = ["orange", "red", "purple", "brown"]
    for w, c in zip(windows, colors):
        ma = df[close_col].rolling(w).mean()
        axes[0].plot(df.index, ma, linewidth=1.2, label=f"MA-{w}h", color=c)
    axes[0].set_title("Close Price with Moving Averages", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Rolling volatility (std of log-returns)
    log_ret = np.log(df[close_col] / df[close_col].shift(1))
    for w, c in zip(windows, colors):
        vol = log_ret.rolling(w).std()
        axes[1].plot(df.index, vol, linewidth=0.9, label=f"Vol-{w}h", color=c)
    axes[1].set_title("Rolling Volatility (Std of Log-Returns)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Volatility")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5) Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    figsize: tuple[int, int] = (12, 10),
    title: str = "Correlation Matrix",
) -> None:
    """Annotated heatmap of the correlation matrix.

    Parameters
    ----------
    columns : list[str] | None
        Columns to include. ``None`` → all numeric columns.
    """
    _apply_style()
    if columns is None:
        subset = df.select_dtypes(include=[np.number])
    else:
        subset = df[columns]

    corr = subset.corr()

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6) Seasonal / time-of-day patterns
# ---------------------------------------------------------------------------


def plot_hourly_seasonality(
    df: pd.DataFrame,
    close_col: str = "Close",
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Show average return and volatility by hour-of-day and day-of-week."""
    _apply_style()
    log_ret = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    ret_df = pd.DataFrame({"log_ret": log_ret})
    ret_df["hour"] = ret_df.index.hour
    ret_df["dow"] = ret_df.index.dayofweek

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Mean return by hour
    hourly_mean = ret_df.groupby("hour")["log_ret"].mean()
    axes[0].bar(hourly_mean.index, hourly_mean.values, color="steelblue", edgecolor="black")
    axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_title("Mean Log-Return by Hour", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Hour (UTC)")
    axes[0].set_ylabel("Mean Log Return")

    # Volatility by hour
    hourly_vol = ret_df.groupby("hour")["log_ret"].std()
    axes[1].bar(hourly_vol.index, hourly_vol.values, color="orange", edgecolor="black")
    axes[1].set_title("Volatility by Hour", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel("Std of Log Return")

    # Mean return by day of week
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_mean = ret_df.groupby("dow")["log_ret"].mean()
    axes[2].bar(dow_mean.index, dow_mean.values, color="mediumseagreen", edgecolor="black")
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(dow_labels)
    axes[2].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[2].set_title("Mean Log-Return by Day of Week", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Mean Log Return")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7) Outlier highlight plot
# ---------------------------------------------------------------------------


def plot_outliers(
    df: pd.DataFrame,
    close_col: str = "Close",
    iqr_factor: float = 3.0,
    figsize: tuple[int, int] = (16, 6),
) -> None:
    """Plot close price and highlight outlier returns (IQR method).

    Parameters
    ----------
    iqr_factor : float
        Multiplier for the IQR to define the outlier threshold.
    """
    _apply_style()
    returns = df[close_col].pct_change().dropna()
    q1 = returns.quantile(0.25)
    q3 = returns.quantile(0.75)
    iqr = q3 - q1
    mask = (returns < q1 - iqr_factor * iqr) | (returns > q3 + iqr_factor * iqr)
    outlier_idx = returns[mask].index

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[close_col], linewidth=0.6, alpha=0.8, label="Close")
    ax.scatter(
        outlier_idx,
        df.loc[outlier_idx, close_col],
        color="red",
        s=12,
        zorder=5,
        label=f"Outliers ({len(outlier_idx)})",
    )
    ax.set_title(
        f"Close Price with Outlier Returns (IQR × {iqr_factor})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  Outliers found: {len(outlier_idx)} / {len(returns)} ({len(outlier_idx)/len(returns)*100:.2f}%)")


# ---------------------------------------------------------------------------
# 8) Convenience: run all plots at once
# ---------------------------------------------------------------------------


def plot_full_eda(
    df: pd.DataFrame,
    title_prefix: str = "Bitcoin",
    close_col: str = "Close",
) -> None:
    """Run every visualisation above in sequence — quick full EDA."""
    plot_price_volume_range(df, title_prefix=title_prefix)
    plot_ohlc(df, title=f"{title_prefix} OHLC (last 7 days)")
    plot_returns_distribution(df, close_col=close_col)
    plot_rolling_statistics(df, close_col=close_col)
    plot_correlation_heatmap(df)
    plot_hourly_seasonality(df, close_col=close_col)
    plot_outliers(df, close_col=close_col)