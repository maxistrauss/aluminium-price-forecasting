from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

try:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger

from timeseries_forecast.eval.eval import evaluate
from timeseries_forecast.experiments.darts_dl.models import MODEL_ORDER, ModelSpec, build_model_specs
from timeseries_forecast.experiments.darts_dl.pipeline import DartsPreparedData, prepare_darts_dataset


@dataclass
class DartsExperimentConfig:
    frequency_resample: str = "24h"
    forecast_horizon: int = 1
    lag_window_len: int = 24
    target_col: str = "BTC_Close_log_ret"
    include_contemporaneous_eth_log_ret: bool = True
    input_chunk_length: int = 24
    output_chunk_length: int = 1
    n_epochs: int = 30
    batch_size: int = 64
    random_state: int = 42
    use_gpu: bool = False
    normalize_data: bool = True
    eval_mode: str = "walk_forward_1step"
    selected_models: Tuple[str, ...] | None = None
    plot_loss_curves: bool = True
    live_plot_loss_curves: bool = False
    results_dir: str = "results/darts_deep_models"


class _LiveLossPlotCallback(Callback):
    def __init__(self, model_name: str, refresh_every_n_epochs: int = 1):
        super().__init__()
        self.model_name = model_name
        self.refresh_every_n_epochs = max(1, refresh_every_n_epochs)
        self.epochs: List[int] = []
        self.train_losses: List[float | None] = []
        self.val_losses: List[float | None] = []

    @staticmethod
    def _metric_to_float(metric_value) -> float | None:
        if metric_value is None:
            return None
        if hasattr(metric_value, "item"):
            return float(metric_value.item())
        try:
            return float(metric_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_metric(callback_metrics: Dict[str, object], candidates: List[str]) -> float | None:
        for metric_name in candidates:
            if metric_name in callback_metrics:
                metric_val = _LiveLossPlotCallback._metric_to_float(callback_metrics[metric_name])
                if metric_val is not None:
                    return metric_val
        return None

    def _render_plot(self) -> None:
        try:
            from IPython.display import clear_output, display
        except ImportError:
            return

        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 4))

        train_points = [(e, l) for e, l in zip(self.epochs, self.train_losses) if l is not None]
        val_points = [(e, l) for e, l in zip(self.epochs, self.val_losses) if l is not None]

        if train_points:
            ax.plot([e for e, _ in train_points], [l for _, l in train_points], label="Train Loss")
        if val_points:
            ax.plot([e for e, _ in val_points], [l for _, l in val_points], label="Val Loss")

        ax.set_title(f"{self.model_name} - Live Train/Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        display(fig)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        callback_metrics: Dict[str, object] = dict(trainer.callback_metrics)
        train_loss = self._get_metric(callback_metrics, ["train_loss", "train_loss_epoch"])
        val_loss = self._get_metric(callback_metrics, ["val_loss", "val_loss_epoch"])

        epoch = int(trainer.current_epoch)
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if (epoch + 1) % self.refresh_every_n_epochs == 0:
            self._render_plot()

    def on_fit_end(self, trainer, pl_module) -> None:
        if self.epochs:
            self._render_plot()


def _get_feature_mode_label(include_contemporaneous_eth_log_ret: bool) -> str:
    return "leaky_contemporaneous_eth_log_ret" if include_contemporaneous_eth_log_ret else "strict_no_contemporaneous"


def _to_timeseries(data: DartsPreparedData) -> Dict[str, TimeSeries]:
    train_target = TimeSeries.from_series(data.train_df[data.target_col])
    val_target = TimeSeries.from_series(data.val_df[data.target_col])
    test_target = TimeSeries.from_series(data.test_df[data.target_col])

    train_cov = TimeSeries.from_dataframe(data.train_df[data.covariate_cols])
    val_cov = TimeSeries.from_dataframe(data.val_df[data.covariate_cols])
    test_cov = TimeSeries.from_dataframe(data.test_df[data.covariate_cols])

    return {
        "train_target": train_target,
        "val_target": val_target,
        "test_target": test_target,
        "train_cov": train_cov,
        "val_cov": val_cov,
        "test_cov": test_cov,
    }


def _maybe_scale_timeseries(
    ts_data: Dict[str, TimeSeries],
    normalize_data: bool,
) -> Tuple[Dict[str, TimeSeries], Scaler | None]:
    if not normalize_data:
        return ts_data, None

    target_scaler = Scaler()
    covariate_scaler = Scaler()

    train_target_scaled = target_scaler.fit_transform(ts_data["train_target"])
    val_target_scaled = target_scaler.transform(ts_data["val_target"])
    test_target_scaled = target_scaler.transform(ts_data["test_target"])

    train_cov_scaled = covariate_scaler.fit_transform(ts_data["train_cov"])
    val_cov_scaled = covariate_scaler.transform(ts_data["val_cov"])
    test_cov_scaled = covariate_scaler.transform(ts_data["test_cov"])

    return (
        {
            "train_target": train_target_scaled,
            "val_target": val_target_scaled,
            "test_target": test_target_scaled,
            "train_cov": train_cov_scaled,
            "val_cov": val_cov_scaled,
            "test_cov": test_cov_scaled,
        },
        target_scaler,
    )


def _evaluate_forecast(y_true: pd.Series, y_pred: pd.Series, target_col: str):
    eval_df = pd.DataFrame({target_col: y_true.values, "pred": y_pred.values}, index=y_true.index)
    return evaluate(
        eval_df,
        target_col=target_col,
        y_pred_col="pred",
        do_calculate_directed_accuracy=True,
    )


def _timeseries_to_pd_series(ts: TimeSeries) -> pd.Series:
    if hasattr(ts, "to_series"):
        return ts.to_series()
    if hasattr(ts, "pd_series"):
        return ts.pd_series()
    raise AttributeError("Unsupported Darts TimeSeries version: missing to_series()/pd_series().")


def _resolve_model_names(selected_models: Tuple[str, ...] | None) -> List[str]:
    if selected_models is None:
        return list(MODEL_ORDER)

    deduped_model_names: List[str] = list(dict.fromkeys(selected_models))
    unknown_models = [name for name in deduped_model_names if name not in MODEL_ORDER]
    if unknown_models:
        raise ValueError(
            f"Unknown model names in selected_models: {unknown_models}. "
            f"Available: {MODEL_ORDER}"
        )

    return deduped_model_names


def _extract_loss_curves(metrics_path: Path, model_name: str) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()

    metrics_df = pd.read_csv(metrics_path)
    if "epoch" not in metrics_df.columns:
        return pd.DataFrame()

    train_col = None
    for candidate in ["train_loss_epoch", "train_loss"]:
        if candidate in metrics_df.columns:
            train_col = candidate
            break

    val_col = None
    for candidate in ["val_loss", "val_loss_epoch"]:
        if candidate in metrics_df.columns:
            val_col = candidate
            break

    if train_col is None and val_col is None:
        return pd.DataFrame()

    loss_df = pd.DataFrame({"epoch": metrics_df["epoch"]})
    if train_col is not None:
        loss_df["train_loss"] = metrics_df[train_col]
    else:
        loss_df["train_loss"] = pd.NA

    if val_col is not None:
        loss_df["val_loss"] = metrics_df[val_col]
    else:
        loss_df["val_loss"] = pd.NA

    loss_df = (
        loss_df.groupby("epoch", as_index=False)
        .last()
        .dropna(subset=["train_loss", "val_loss"], how="all")
        .reset_index(drop=True)
    )

    if loss_df.empty:
        return pd.DataFrame()

    loss_df["model"] = model_name
    return loss_df[["model", "epoch", "train_loss", "val_loss"]]


def _save_loss_plot(loss_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    if loss_df.empty:
        return

    plt.figure(figsize=(8, 4))
    if loss_df["train_loss"].notna().any():
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
    if loss_df["val_loss"].notna().any():
        plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss")

    plt.title(f"{model_name} - Train/Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _one_step_walk_forward_forecast(
    model,
    covariate_type: str,
    history_target: TimeSeries,
    forecast_target: TimeSeries,
    all_covariates: TimeSeries,
) -> TimeSeries:
    full_series = history_target.append(forecast_target)

    if covariate_type == "future":
        return model.historical_forecasts(
            series=full_series,
            future_covariates=all_covariates,
            start=forecast_target.start_time(),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            last_points_only=True,
            verbose=True,
        )

    return model.historical_forecasts(
        series=full_series,
        past_covariates=all_covariates,
        start=forecast_target.start_time(),
        forecast_horizon=1,
        stride=1,
        retrain=False,
        last_points_only=True,
        verbose=True,
    )


def _run_single_model(
    model_spec: ModelSpec,
    ts_data_train: Dict[str, TimeSeries],
    ts_data_eval: Dict[str, TimeSeries],
    target_col: str,
    feature_mode: str,
    target_scaler: Scaler | None,
    eval_mode: str,
) -> Tuple[Dict[str, float | str], pd.DataFrame]:
    if model_spec.availability != "available" or model_spec.model is None:
        result_row: Dict[str, float | str] = {
            "Model": model_spec.name,
            "Feature_Mode": feature_mode,
            "Status": model_spec.availability,
            "Note": model_spec.note,
        }
        return result_row, pd.DataFrame()

    model = model_spec.model

    train_target = ts_data_train["train_target"]
    val_target = ts_data_train["val_target"]
    test_target = ts_data_train["test_target"]

    train_cov = ts_data_train["train_cov"]
    val_cov = ts_data_train["val_cov"]
    test_cov = ts_data_train["test_cov"]

    all_cov_train_val = train_cov.append(val_cov)
    all_cov_full = all_cov_train_val.append(test_cov)

    if model_spec.covariate_type == "future":
        model.fit(
            series=train_target,
            future_covariates=train_cov,
            val_series=val_target,
            val_future_covariates=val_cov,
            verbose=True,
        )
    else:
        model.fit(
            series=train_target,
            past_covariates=train_cov,
            val_series=val_target,
            val_past_covariates=val_cov,
            verbose=True,
        )

    if eval_mode == "walk_forward_1step":
        val_prediction = _one_step_walk_forward_forecast(
            model=model,
            covariate_type=model_spec.covariate_type,
            history_target=train_target,
            forecast_target=val_target,
            all_covariates=all_cov_train_val,
        )
        test_prediction = _one_step_walk_forward_forecast(
            model=model,
            covariate_type=model_spec.covariate_type,
            history_target=train_target.append(val_target),
            forecast_target=test_target,
            all_covariates=all_cov_full,
        )
    else:
        val_len = len(val_target)
        test_len = len(test_target)
        if model_spec.covariate_type == "future":
            val_prediction = model.predict(
                n=val_len,
                series=train_target,
                future_covariates=all_cov_train_val,
                verbose=True,
            )

            test_prediction = model.predict(
                n=test_len,
                series=train_target.append(val_target),
                future_covariates=all_cov_full,
                verbose=True,
            )
        else:
            val_prediction = model.predict(
                n=val_len,
                series=train_target,
                past_covariates=all_cov_train_val,
                verbose=True,
            )

            test_prediction = model.predict(
                n=test_len,
                series=train_target.append(val_target),
                past_covariates=all_cov_full,
                verbose=True,
            )

    if target_scaler is not None:
        val_prediction = target_scaler.inverse_transform(val_prediction)
        test_prediction = target_scaler.inverse_transform(test_prediction)

    val_pred_series = _timeseries_to_pd_series(val_prediction)
    test_pred_series = _timeseries_to_pd_series(test_prediction)
    val_true_series = _timeseries_to_pd_series(ts_data_eval["val_target"])
    test_true_series = _timeseries_to_pd_series(ts_data_eval["test_target"])

    val_eval = _evaluate_forecast(val_true_series, val_pred_series, target_col=target_col)
    test_eval = _evaluate_forecast(test_true_series, test_pred_series, target_col=target_col)

    result_row = {
        "Model": model_spec.name,
        "Feature_Mode": feature_mode,
        "Status": "available",
        "Test_MAE": test_eval.mae,
        "Test_RMSE": test_eval.rmse,
        "Test_MSE": test_eval.mse,
        "Test_R2": test_eval.r2,
        "Test_SMAPE": test_eval.smape,
        "Test_SMAPE_Total": test_eval.smape_total,
        "Test_Forecast_Bias": test_eval.forecast_bias,
        "Test_Accuracy_Total": test_eval.accuracy_total,
        "Test_Directed_Accuracy": test_eval.directed_accuracy,
    }

    prediction_df = pd.DataFrame(
        {
            "timestamp": test_true_series.index,
            "model": model_spec.name,
            "feature_mode": feature_mode,
            "y_true": test_true_series.values,
            "y_pred": test_pred_series.values,
        }
    )

    return result_row, prediction_df


def run_darts_experiments(config: DartsExperimentConfig) -> pd.DataFrame:
    feature_mode = _get_feature_mode_label(config.include_contemporaneous_eth_log_ret)
    selected_model_names = _resolve_model_names(config.selected_models)
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    leaky_covariate_col = "ETH_Close_log_ret_leaky_t"

    prepared_data = prepare_darts_dataset(
        frequency_resample=config.frequency_resample,
        forecast_horizon=config.forecast_horizon,
        lag_window_len=config.lag_window_len,
        target_col=config.target_col,
        include_contemporaneous_eth_log_ret=config.include_contemporaneous_eth_log_ret,
    )
    print("\n[DEBUG] Covariate columns:")
    print(prepared_data.covariate_cols)
    print(
        f"[DEBUG] Leaky covariate '{leaky_covariate_col}' present: "
        f"{leaky_covariate_col in prepared_data.covariate_cols}"
    )

    ts_data_raw = _to_timeseries(prepared_data)
    ts_data_train, target_scaler = _maybe_scale_timeseries(
        ts_data=ts_data_raw,
        normalize_data=config.normalize_data,
    )

    result_rows: List[Dict[str, float | str]] = []
    all_predictions: List[pd.DataFrame] = []
    all_loss_curves: List[pd.DataFrame] = []

    for model_name in selected_model_names:
        print(f"\n{'=' * 90}")
        print(f"Running model: {model_name}")
        print(f"{'=' * 90}")
        print(f"[DEBUG] {model_name} uses covariates: {prepared_data.covariate_cols}")
        print(
            f"[DEBUG] {model_name} leaky covariate active: "
            f"{leaky_covariate_col in prepared_data.covariate_cols}"
        )

        trainer_overrides: Dict[str, object] = {}
        metrics_path = None

        if config.plot_loss_curves:
            model_logger = CSVLogger(
                save_dir=str(results_path / "training_logs"),
                name=model_name,
                version="metrics",
            )
            trainer_overrides["logger"] = model_logger
            metrics_path = Path(model_logger.log_dir) / "metrics.csv"

        if config.live_plot_loss_curves:
            trainer_overrides["callbacks"] = [_LiveLossPlotCallback(model_name=model_name)]

        model_specs = build_model_specs(
            input_chunk_length=config.input_chunk_length,
            output_chunk_length=config.output_chunk_length,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            random_state=config.random_state,
            use_gpu=config.use_gpu,
            selected_model_names=[model_name],
            extra_trainer_kwargs=trainer_overrides if trainer_overrides else None,
        )

        result_row, prediction_df = _run_single_model(
            model_spec=model_specs[model_name],
            ts_data_train=ts_data_train,
            ts_data_eval=ts_data_raw,
            target_col=config.target_col,
            feature_mode=feature_mode,
            target_scaler=target_scaler,
            eval_mode=config.eval_mode,
        )
        result_rows.append(result_row)

        if not prediction_df.empty:
            all_predictions.append(prediction_df)

        if config.plot_loss_curves and metrics_path is not None:
            loss_df = _extract_loss_curves(metrics_path, model_name)
            if not loss_df.empty:
                all_loss_curves.append(loss_df)
                plot_path = results_path / "plots" / f"{model_name.lower()}_loss_curve.png"
                _save_loss_plot(loss_df, model_name=model_name, output_path=plot_path)

    results_df = pd.DataFrame(result_rows)

    config_path = results_path / "experiment_config.csv"
    pd.DataFrame([asdict(config)]).to_csv(config_path, index=False)

    metrics_path = results_path / "darts_deep_models_results.csv"
    results_df.to_csv(metrics_path, index=False)

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = results_path / "darts_deep_models_test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

    if all_loss_curves:
        loss_curves_df = pd.concat(all_loss_curves, ignore_index=True)
        loss_curves_path = results_path / "darts_deep_models_loss_curves.csv"
        loss_curves_df.to_csv(loss_curves_path, index=False)
        print(f"Saved loss curves to: {loss_curves_path}")

    print("\nExperiment finished.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved config to: {config_path}")

    return results_df

