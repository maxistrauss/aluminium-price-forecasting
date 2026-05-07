import numpy as np
import pandas as pd
from typing import Optional

def set_timestamp_as_index(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], dayfirst=True, errors="coerce")
    df = df[df[timestamp_column].notna()].copy()
    df.set_index(timestamp_column, inplace=True)
    df.sort_index(inplace=True)
    return df


def interpolate_gaps(
    df: pd.DataFrame,
    freq: str = "h",
    exclude_cols: Optional[list[str]] = None,
    datetime_cols: Optional[list[str]] = None,
    zero_as_nan_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Fill time-series gaps via reindexing and linear interpolation.

    Steps
    -----
    1. Remove rows whose index is NaT.
    2. Drop duplicate index entries (keeps the first occurrence).
    3. Create a complete ``DatetimeIndex`` at the given *freq* and reindex.
    4. For every numeric column (except those in *exclude_cols*):
       linear interpolation → forward fill → backward fill.
    5. For columns listed in *datetime_cols*: linear interpolation of the
       underlying datetime values.
    6. For columns listed in *zero_as_nan_cols*: replace 0 with NaN
       before interpolation (e.g. ``["Volume", "Close"]``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``.
    freq : str, optional
        Frequency string for ``pd.date_range``, by default ``"h"`` (hourly).
    exclude_cols : list[str] | None, optional
        Numeric columns to skip during interpolation (e.g. ``["Ignore"]``).
    datetime_cols : list[str] | None, optional
        Non-numeric datetime columns to interpolate (e.g. ``["Close time"]``).
    zero_as_nan_cols : list[str] | None, optional
        Columns where 0 should be treated as missing and replaced with NaN
        before interpolation (e.g. ``["Volume", "Number of trades"]``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a complete time index and interpolated values.
    """
    if exclude_cols is None:
        exclude_cols = []
    if datetime_cols is None:
        datetime_cols = []
    if zero_as_nan_cols is None:
        zero_as_nan_cols = []

    print("\n" + "=" * 80)
    print("Data Cleaning – Interpolation von Missing Values und Nullwerten")
    print("=" * 80)

    # 1) Remove NaT index rows
    df_clean = df[df.index.notna()].copy()

    # 2) Remove duplicate index entries
    n_dup = df_clean.index.duplicated(keep="first").sum()
    if n_dup > 0:
        print(f"  {n_dup} doppelte Index-Einträge entfernt")
    df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

    # 3) Reindex to a complete range
    full_index = pd.date_range(
        start=df_clean.index.min(), end=df_clean.index.max(), freq=freq
    )
    n_missing = len(full_index) - len(df_clean)
    print(f"\n  Vollständiger Zeitindex: {len(full_index):,} Einträge")
    print(f"  Aktueller DataFrame:    {len(df_clean):,} Zeilen")
    print(f"  Zu ergänzende Einträge: {n_missing:,}")

    df_filled = df_clean.reindex(full_index)

    # 3b) Replace zeros with NaN in specified columns
    for col in zero_as_nan_cols:
        if col not in df_filled.columns:
            continue
        n_zeros = int((df_filled[col] == 0).sum())
        if n_zeros > 0:
            df_filled[col] = df_filled[col].replace(0, np.nan)
            print(f"    {col}: {n_zeros} Nullwerte -> NaN ersetzt")

    # 4) Interpolate numeric columns
    numeric_cols = [
        c
        for c in df_filled.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    print(f"\n  Interpoliere {len(numeric_cols)} numerische Spalten:")
    for col in numeric_cols:
        nan_before = int(df_filled[col].isnull().sum())
        if nan_before > 0:
            df_filled[col] = (
                df_filled[col]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
            nan_after = int(df_filled[col].isnull().sum())
            print(f"    {col}: {nan_before} -> {nan_after} NaN")
        else:
            print(f"    {col}: keine NaN")

    # 5) Interpolate datetime columns
    for col in datetime_cols:
        if col not in df_filled.columns:
            continue
        nan_count = int(df_filled[col].isnull().sum())
        if nan_count > 0:
            df_filled[col] = df_filled[col].interpolate(
                method="linear", limit_direction="both"
            )
            print(f"    {col}: {nan_count} NaN interpoliert")

    print(f"\n  Shape vorher:  {df.shape}")
    print(f"  Shape nachher: {df_filled.shape}")
    print("OK: Interpolation abgeschlossen!")

    return df_filled