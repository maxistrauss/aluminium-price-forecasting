"""
Walk-Forward Validation (expanding window).

Each fold expands the training window by step_size, validation window follows
immediately after — no gap, no leakage.

Diagram:
  Fold 0: [===== TRAIN =====][VAL]
  Fold 1: [======= TRAIN =======][VAL]
  Fold 2: [========= TRAIN =========][VAL]
  ...
  Final holdout: [============ TRAIN ============][TEST]
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import pandas as pd


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_size: int
    val_size: int

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_id:2d}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}] ({self.train_size} rows) | "
            f"val  [{self.val_start.date()} → {self.val_end.date()}] ({self.val_size} rows)"
        )


class WalkForwardSplitter:
    """
    Expanding-window walk-forward splitter.

    Parameters
    ----------
    initial_train_size : int
        Minimum number of rows in the first training fold.
    step_size : int
        How many rows to advance the training window each fold.
    val_size : int
        Number of rows in each validation window.
    """

    def __init__(
        self,
        initial_train_size: int = 756,
        step_size: int = 63,
        val_size: int = 63,
    ):
        if initial_train_size <= 0:
            raise ValueError("initial_train_size must be > 0")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if val_size <= 0:
            raise ValueError("val_size must be > 0")

        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.val_size = val_size

    # ------------------------------------------------------------------
    # Core splitter
    # ------------------------------------------------------------------

    def get_folds(self, df: pd.DataFrame) -> List[WalkForwardFold]:
        """
        Return all walk-forward folds for *df*.

        The DataFrame must be sorted by its DatetimeIndex (ascending).
        Only rows up to the last row that still leaves a full val window
        are used.
        """
        n = len(df)
        min_required = self.initial_train_size + self.val_size
        if n < min_required:
            raise ValueError(
                f"DataFrame has {n} rows but at least {min_required} are needed "
                f"(initial_train_size={self.initial_train_size} + val_size={self.val_size})."
            )

        folds: List[WalkForwardFold] = []
        fold_id = 0
        train_end_pos = self.initial_train_size

        while train_end_pos + self.val_size <= n:
            val_end_pos = train_end_pos + self.val_size

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=df.index[0],
                    train_end=df.index[train_end_pos - 1],
                    val_start=df.index[train_end_pos],
                    val_end=df.index[val_end_pos - 1],
                    train_size=train_end_pos,
                    val_size=self.val_size,
                )
            )
            fold_id += 1
            train_end_pos += self.step_size

        return folds

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[WalkForwardFold, pd.DataFrame, pd.DataFrame]]:
        """
        Yield (fold, train_df, val_df) for each walk-forward fold.

        Both DataFrames share the same columns as *df*.
        """
        for fold in self.get_folds(df):
            train_df = df.loc[: fold.train_end].copy()
            val_df = df.loc[fold.val_start : fold.val_end].copy()
            yield fold, train_df, val_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def n_folds(self, n: int) -> int:
        """Number of folds for a dataset of *n* rows."""
        if n < self.initial_train_size + self.val_size:
            return 0
        return (n - self.initial_train_size - self.val_size) // self.step_size + 1

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print a human-readable summary of all folds."""
        folds = self.get_folds(df)
        print("=" * 72)
        print("Walk-Forward Validation Summary")
        print("=" * 72)
        print(f"  Dataset : {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")
        print(f"  Initial train size : {self.initial_train_size}")
        print(f"  Step size          : {self.step_size}")
        print(f"  Val size           : {self.val_size}")
        print(f"  Number of folds    : {len(folds)}")
        print()
        for fold in folds:
            print(f"  {fold}")
        print("=" * 72)
