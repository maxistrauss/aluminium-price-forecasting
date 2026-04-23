from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from darts.models import NBEATSModel, NHiTSModel, RNNModel, TransformerModel
except ImportError as exc:
    raise ImportError(
        "Darts is required for darts_dl experiments. Install with: pip install 'darts[torch]'"
    ) from exc


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: Optional[Any]
    covariate_type: str
    availability: str
    note: str = ""


MODEL_ORDER: List[str] = [
    "Transformer",
    "LSTM",
    "GRU",
    "NBEATS",
    "NHITS",
    "INformer",
    "Autoformer",
]


def _base_trainer_kwargs(use_gpu: bool, extra_trainer_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    trainer_kwargs: Dict[str, Any] = {
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1,
    }
    if extra_trainer_kwargs:
        trainer_kwargs.update(extra_trainer_kwargs)
    return trainer_kwargs


def build_model_specs(
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int,
    batch_size: int,
    random_state: int,
    use_gpu: bool,
    selected_model_names: Optional[List[str]] = None,
    extra_trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, ModelSpec]:
    """Build all requested model definitions and metadata.

    `covariate_type` values:
    - "past": pass covariates via `past_covariates`
    - "future": pass covariates via `future_covariates`
    """
    common_kwargs = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "random_state": random_state,
        "pl_trainer_kwargs": _base_trainer_kwargs(use_gpu, extra_trainer_kwargs),
        "force_reset": True,
        "save_checkpoints": False,
    }

    specs: Dict[str, ModelSpec] = {
        "Transformer": ModelSpec(
            name="Transformer",
            model=TransformerModel(
                **common_kwargs,
                d_model=64,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=256,
                dropout=0.1,
            ),
            covariate_type="past",
            availability="available",
        ),
        "LSTM": ModelSpec(
            name="LSTM",
            model=RNNModel(
                model="LSTM",
                input_chunk_length=input_chunk_length,
                training_length=max(input_chunk_length * 2, input_chunk_length + 1),
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=random_state,
                pl_trainer_kwargs=_base_trainer_kwargs(use_gpu, extra_trainer_kwargs),
                force_reset=True,
                save_checkpoints=False,
            ),
            covariate_type="future",
            availability="available",
        ),
        "GRU": ModelSpec(
            name="GRU",
            model=RNNModel(
                model="GRU",
                input_chunk_length=input_chunk_length,
                training_length=max(input_chunk_length * 2, input_chunk_length + 1),
                n_epochs=n_epochs,
                batch_size=batch_size,
                random_state=random_state,
                pl_trainer_kwargs=_base_trainer_kwargs(use_gpu, extra_trainer_kwargs),
                force_reset=True,
                save_checkpoints=False,
            ),
            covariate_type="future",
            availability="available",
        ),
        "NBEATS": ModelSpec(
            name="NBEATS",
            model=NBEATSModel(
                **common_kwargs,
                num_stacks=12,
                num_blocks=1,
                num_layers=4,
                layer_widths=256,
                dropout=0.1,
            ),
            covariate_type="past",
            availability="available",
        ),
        "NHITS": ModelSpec(
            name="NHITS",
            model=NHiTSModel(
                **common_kwargs,
                num_stacks=3,
                num_blocks=1,
                num_layers=2,
                layer_widths=256,
                dropout=0.1,
            ),
            covariate_type="past",
            availability="available",
        ),
        "INformer": ModelSpec(
            name="INformer",
            model=None,
            covariate_type="past",
            availability="unavailable",
            note="InformerModel is not available in current Darts forecasting model registry.",
        ),
        "Autoformer": ModelSpec(
            name="Autoformer",
            model=None,
            covariate_type="past",
            availability="unavailable",
            note="AutoformerModel is not available in current Darts forecasting model registry.",
        ),
    }

    if not selected_model_names:
        return specs

    return {name: specs[name] for name in selected_model_names if name in specs}
