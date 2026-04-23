from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DuplicateCheckResult:
    """Result of a duplicate check on a DataFrame."""

    has_index_duplicates: bool
    index_duplicate_count: int
    has_row_duplicates: bool
    row_duplicate_count: int


def check_duplicates(
    df: pd.DataFrame, index_name: str = "Zeitstempel"
) -> DuplicateCheckResult:
    """Check a DataFrame for index duplicates and full row duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check. Its index is used for the timestamp
        duplicate check.
    index_name : str, optional
        Label used in the printed output to describe the index,
        by default ``"Zeitstempel"``.

    Returns
    -------
    DuplicateCheckResult
        Flags and counts for index and full-row duplicates.
    """
    print("\n" + "=" * 80)
    print("Duplikat-Check")
    print("=" * 80)

    # 1) Duplicates based on the index
    has_index_duplicates = df.index.has_duplicates
    index_duplicate_count = 0
    if has_index_duplicates:
        dup_idx = df.index[df.index.duplicated(keep=False)]
        index_duplicate_count = len(dup_idx)
        print(f"⚠ {index_duplicate_count} Einträge mit doppeltem {index_name}")
        print("  Beispiele (erste 10):")
        for ts in dup_idx.unique()[:10]:
            count_ts = (df.index == ts).sum()
            print(f"    {ts} (Anzahl: {count_ts})")
    else:
        print(f"✓ Keine doppelten {index_name} im Index")

    # 2) Full row duplicates (all columns identical)
    dup_rows = df[df.duplicated(keep=False)]
    row_duplicate_count = len(dup_rows)
    has_row_duplicates = row_duplicate_count > 0
    if has_row_duplicates:
        print(f"⚠ {row_duplicate_count} Zeilen sind doppelt (identischer Inhalt)")
        print("  Beispiele (erste 5 Timestamps):")
        print(dup_rows.head().index)
    else:
        print("✓ Keine identischen Zeilenduplikate gefunden")

    return DuplicateCheckResult(
        has_index_duplicates=has_index_duplicates,
        index_duplicate_count=index_duplicate_count,
        has_row_duplicates=has_row_duplicates,
        row_duplicate_count=row_duplicate_count,
    )


def check_missing_values(df: pd.DataFrame, expected_freq: Optional[str] = "h") -> bool:
    # printe Ergebnisse
    print("\n" + "=" * 80)
    print("Fehlende Werte Check")
    print("=" * 80)

    if expected_freq is None:
        print("- Frequenzprüfung übersprungen (expected_freq=None)")
        return False

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
    missing_timestamps = full_index.difference(df.index)
    n_gaps = len(missing_timestamps)

    if n_gaps > 0:
        print(f"⚠ Es gibt {n_gaps} Lücken in der Zeitreihe (erwartete Frequenz: {expected_freq}).")
    else:
        print(f"✓ Keine Lücken in der Zeitreihe gefunden (Frequenz: {expected_freq}).")

    return n_gaps > 0


@dataclass
class NanCheckResult:
    """Result of a NaN check on a DataFrame."""

    has_nan: bool
    total_nan_count: int
    nan_per_column: dict[str, int]


def check_nan_values(df: pd.DataFrame) -> NanCheckResult:
    """Check a DataFrame for NaN values across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    NanCheckResult
        Flags and counts for NaN values per column.
    """
    print("\n" + "=" * 80)
    print("NaN-Check")
    print("=" * 80)

    nan_counts = df.isnull().sum()
    nan_percent = (nan_counts / len(df)) * 100
    total_nan = nan_counts.sum()

    # Only keep columns that actually have NaN values
    nan_per_column = {
        str(col): int(count) for col, count in nan_counts.items() if count > 0
    }

    if total_nan > 0:
        print(f"⚠ {total_nan} NaN-Werte insgesamt gefunden:")
        for col, count in nan_per_column.items():
            pct = nan_percent[col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("✓ Keine NaN-Werte gefunden!")

    return NanCheckResult(
        has_nan=total_nan > 0,
        total_nan_count=int(total_nan),
        nan_per_column=nan_per_column,
    )


@dataclass
class NullCheckResult:
    """Result of a 0-value check on a DataFrame."""

    has_null: bool
    total_null_count: int
    null_per_column: dict[str, int]


def check_0_values(
    df: pd.DataFrame,
    ignore_columns: Optional[list[str]] = None,
) -> NullCheckResult:
    """Check a DataFrame for 0 values across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    NullCheckResult
        Flags and counts for 0 values per column.
    """
    print("\n" + "=" * 80)
    print("0-Werte Check")
    print("=" * 80)

    if ignore_columns is None:
        ignore_columns = []

    cols_to_check = [col for col in df.columns if col not in ignore_columns]
    null_counts = (df[cols_to_check] == 0).sum()
    null_percent = (null_counts / len(df)) * 100
    total_null = null_counts.sum()

    # Only keep columns that actually have 0 values
    null_per_column = {
        str(col): int(count) for col, count in null_counts.items() if count > 0
    }

    if total_null > 0:
        print(f"⚠ {total_null} 0-Werte insgesamt gefunden:")
        for col, count in null_per_column.items():
            pct = null_percent[col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("✓ Keine 0-Werte gefunden!")

    return NullCheckResult(
        has_null=total_null > 0,
        total_null_count=int(total_null),
        null_per_column=null_per_column,
    )


