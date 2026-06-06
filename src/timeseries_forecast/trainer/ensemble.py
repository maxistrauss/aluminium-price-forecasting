"""
Ensemble + Uncertainty estimation for multi-task forecasting.

Two complementary approaches:

1. Bootstrap Ensemble
   Train N independent models on bootstrap resamples of the training data.
   Aggregate predictions → mean (point estimate) + std (uncertainty).

2. MC-Dropout (already in MultiTaskTrainer.predict_with_uncertainty)
   This module adds a standalone wrapper usable with any sklearn-style model.

Evaluation helpers
------------------
coverage_rate(y_true, mean, std, z=1.96)  →  fraction inside 95% CI
interval_width(std, z=1.96)               →  average CI width
crps_gaussian(y_true, mean, std)          →  CRPS for Gaussian predictive dist
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Bootstrap Ensemble (sklearn-style models)
# ---------------------------------------------------------------------------

class BootstrapEnsemble:
    """
    Bootstrap ensemble for any sklearn-compatible estimator.

    Parameters
    ----------
    base_model : unfitted sklearn estimator (will be cloned N times).
    n_models   : number of bootstrap models.
    sample_frac: fraction of training data per bootstrap resample.
    random_state: seed for reproducibility (each model gets seed + i).
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        n_models: int = 10,
        sample_frac: float = 0.8,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.n_models = n_models
        self.sample_frac = sample_frac
        self.random_state = random_state
        self._models: List[BaseEstimator] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> "BootstrapEnsemble":
        """Fit N models on bootstrap resamples of (X_train, y_train)."""
        self._models = []
        n_samples = max(1, int(len(X_train) * self.sample_frac))

        it = range(self.n_models)
        if verbose:
            it = tqdm(it, desc="Bootstrap ensemble", unit="model")

        for i in it:
            rng = self.random_state + i
            X_b, y_b = resample(X_train, y_train, n_samples=n_samples, random_state=rng)
            m = clone(self.base_model)
            # Pass random_state if model supports it
            if hasattr(m, "random_state"):
                m.random_state = rng
            m.fit(X_b, y_b)
            self._models.append(m)

        return self

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_prediction, std_prediction) across ensemble members.
        """
        if not self._models:
            raise RuntimeError("Call fit() before predict().")

        preds = np.stack([m.predict(X) for m in self._models], axis=0)  # (N, T)
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_quantiles(
        self, X: np.ndarray, quantiles: Tuple[float, ...] = (0.025, 0.5, 0.975)
    ) -> np.ndarray:
        """Return empirical quantiles across ensemble members. Shape: (len(quantiles), T)."""
        if not self._models:
            raise RuntimeError("Call fit() before predict_quantiles().")

        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        return np.quantile(preds, quantiles, axis=0)


# ---------------------------------------------------------------------------
# Walk-Forward Ensemble (trains one ensemble per WFV fold)
# ---------------------------------------------------------------------------

class WalkForwardEnsembleResult:
    """Stores per-fold and aggregated predictions with uncertainty."""

    def __init__(self):
        self.fold_results: List[dict] = []
        self.all_dates: List[pd.Timestamp] = []
        self.all_means: List[float] = []
        self.all_stds: List[float] = []
        self.all_targets: List[float] = []

    def add_fold(
        self,
        fold_id: int,
        dates: pd.DatetimeIndex,
        mean_pred: np.ndarray,
        std_pred: np.ndarray,
        y_true: np.ndarray,
    ):
        self.fold_results.append(
            dict(fold_id=fold_id, dates=dates, mean=mean_pred, std=std_pred, y_true=y_true)
        )
        self.all_dates.extend(dates.tolist())
        self.all_means.extend(mean_pred.tolist())
        self.all_stds.extend(std_pred.tolist())
        self.all_targets.extend(y_true.tolist())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.all_dates,
                "mean": self.all_means,
                "std": self.all_stds,
                "y_true": self.all_targets,
                "lower_95": np.array(self.all_means) - 1.96 * np.array(self.all_stds),
                "upper_95": np.array(self.all_means) + 1.96 * np.array(self.all_stds),
            }
        ).set_index("date").sort_index()


def run_walk_forward_bootstrap(
    X: pd.DataFrame,
    y: pd.Series,
    base_model: BaseEstimator,
    splitter,
    n_models: int = 10,
    sample_frac: float = 0.8,
    verbose: bool = True,
) -> WalkForwardEnsembleResult:
    """
    Run bootstrap ensemble over all walk-forward folds.

    Parameters
    ----------
    X        : feature DataFrame (DatetimeIndex, sorted).
    y        : target Series (same index as X).
    base_model : unfitted sklearn estimator.
    splitter : WalkForwardSplitter instance.
    n_models : bootstrap ensemble size.
    sample_frac : fraction of train data per bootstrap resample.
    verbose  : show per-fold progress.

    Returns
    -------
    WalkForwardEnsembleResult with predictions for every validation step.
    """
    result = WalkForwardEnsembleResult()

    folds = splitter.get_folds(X)
    fold_iter = folds
    if verbose:
        fold_iter = tqdm(folds, desc="Walk-forward folds", unit="fold")

    for fold in fold_iter:
        train_X = X.loc[: fold.train_end].dropna()
        train_y = y.loc[: fold.train_end].dropna()
        common_train_idx = train_X.index.intersection(train_y.index)
        train_X = train_X.loc[common_train_idx]
        train_y = train_y.loc[common_train_idx]

        val_X = X.loc[fold.val_start : fold.val_end]
        val_y = y.loc[fold.val_start : fold.val_end]
        val_X_clean = val_X.dropna()
        val_y_clean = val_y.loc[val_X_clean.index]

        if len(train_X) == 0 or len(val_X_clean) == 0:
            continue

        ens = BootstrapEnsemble(
            base_model, n_models=n_models, sample_frac=sample_frac
        )
        ens.fit(train_X.values, train_y.values, verbose=False)
        mean_pred, std_pred = ens.predict(val_X_clean.values)

        result.add_fold(
            fold_id=fold.fold_id,
            dates=val_X_clean.index,
            mean_pred=mean_pred,
            std_pred=std_pred,
            y_true=val_y_clean.values,
        )

    return result


# ---------------------------------------------------------------------------
# Probabilistic Evaluation Metrics
# ---------------------------------------------------------------------------

def coverage_rate(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    z: float = 1.96,
) -> float:
    """
    Fraction of true values falling inside [mean ± z*std].

    For a well-calibrated 95% CI (z=1.96), this should be ~0.95.
    Values << 0.95 → intervals too narrow (overconfident).
    Values >> 0.95 → intervals too wide (underconfident).
    """
    lower = mean - z * std
    upper = mean + z * std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())


def average_interval_width(mean: np.ndarray, std: np.ndarray, z: float = 1.96) -> float:
    """Average width of the prediction interval."""
    return float((2 * z * std).mean())


def crps_gaussian(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.

    Lower is better.  CRPS = 0 for a perfect point forecast.

    Formula: CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ, Φ = normal CDF, φ = normal PDF.
    """
    from scipy import stats  # lazy import

    std_safe = np.clip(std, 1e-8, None)
    z = (y_true - mean) / std_safe
    crps = std_safe * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    return float(crps.mean())


def evaluate_probabilistic(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str = "",
) -> pd.Series:
    """
    Return a pandas Series with all probabilistic evaluation metrics.

    Includes: MAE, RMSE, Coverage@95, AvgWidth, CRPS.
    """
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    metrics = {
        "MAE": mean_absolute_error(y_true, mean),
        "RMSE": root_mean_squared_error(y_true, mean),
        "Coverage_95": coverage_rate(y_true, mean, std),
        "AvgWidth_95": average_interval_width(mean, std),
    }
    try:
        metrics["CRPS"] = crps_gaussian(y_true, mean, std)
    except ImportError:
        metrics["CRPS"] = float("nan")

    name = label if label else "metrics"
    return pd.Series(metrics, name=name)
