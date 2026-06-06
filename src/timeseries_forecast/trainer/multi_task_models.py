"""
Multi-Task + Multi-Horizon forecasting model.

Architecture
------------
Input  : (batch, seq_len, n_features)
           ↓
Shared LSTM encoder  →  last hidden state  (batch, hidden_dim)
           ↓
Shared projection    →  (batch, d_model)
           ↓
Per-(task × horizon) linear heads  →  (batch,) each

This gives one forward pass that simultaneously predicts:
  • all metal log-returns  (multi-task)
  • at horizons h = 1, 5, 10, 21 days  (multi-horizon)

Loss = mean over all active (task, horizon) heads of MSE.
Each head is independent so missing targets (NaN at sequence edges)
are masked out per sample.

Uncertainty
-----------
MC-Dropout: keep dropout active during inference, run N forward passes,
return (mean, std) over those passes.
"""

import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Python 3.11.0rc1 is missing sys.get_int_max_str_digits / set_int_max_str_digits
# which PyTorch ≥2.1 requires through its torch._dynamo polyfill layer.
# The polyfill's substitute_in_graph decorator verifies signature compatibility,
# so the stubs must carry matching type annotations.
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int:  # noqa: E301
        return 4300

    def _set_int_max_str_digits(maxdigits: int) -> None:  # noqa: E301
        pass

    sys.get_int_max_str_digits = _get_int_max_str_digits  # type: ignore[attr-defined]
    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    """Hyper-parameters for MultiTaskMultiHorizonModel."""

    # Architecture
    n_features: int = 0
    tasks: List[str] = field(default_factory=lambda: ["ALU"])
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 21])
    seq_len: int = 21
    hidden_dim: int = 128
    d_model: int = 64
    n_lstm_layers: int = 2
    dropout: float = 0.2

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 100
    patience: int = 15
    min_epochs: int = 30        # never stop before this epoch
    min_delta: float = 1e-6    # minimum val-loss improvement to count as progress
    grad_clip: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        if self.n_features == 0:
            raise ValueError("Set n_features before constructing the model.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """
    Windowed time-series dataset for multi-task multi-horizon learning.

    Parameters
    ----------
    X : (T, n_features) float32 array  — feature matrix
    targets : dict  {(task_name, horizon): (T,) float32 array}
    seq_len : look-back window length
    """

    def __init__(
        self,
        X: np.ndarray,
        targets: Dict[Tuple[str, int], np.ndarray],
        seq_len: int,
    ):
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, n_features), got shape {X.shape}")

        self.X = X.astype(np.float32)
        self.targets = {k: v.astype(np.float32) for k, v in targets.items()}
        self.seq_len = seq_len
        self.max_horizon = max(h for _, h in targets)
        self.T = len(X)

        # Valid start indices: need seq_len history.
        # Targets are pre-shifted (rolling(h).sum().shift(-h)) so target[t]
        # already contains the h-step-ahead cumulative return.
        # NaN targets at the end (no future data) are masked in the loss.
        self.indices = list(range(seq_len, self.T))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        X_seq = torch.from_numpy(self.X[t - self.seq_len : t])  # (seq_len, n_feat)

        y_dict: Dict[Tuple[str, int], torch.Tensor] = {}
        for (task, h), arr in self.targets.items():
            # target[t] = sum of h future log-returns starting at t+1
            val = arr[t] if t < self.T else float("nan")
            y_dict[(task, h)] = torch.tensor(val, dtype=torch.float32)

        return X_seq, y_dict

    @staticmethod
    def collate_fn(batch):
        X_list, y_list = zip(*batch)
        X_batch = torch.stack(X_list)  # (B, seq_len, n_feat)
        keys = y_list[0].keys()
        y_batch = {k: torch.stack([y[k] for y in y_list]) for k in keys}
        return X_batch, y_batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTaskMultiHorizonModel(nn.Module):
    """
    Shared LSTM encoder with independent linear output heads.

    One head per (task, horizon) pair predicts a log-return scalar.
    """

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg

        # Input normalisation
        self.input_norm = nn.LayerNorm(cfg.n_features)

        # Shared LSTM
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(cfg.dropout)

        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Output heads: one per (task, horizon)
        self.heads = nn.ModuleDict(
            {
                self._head_key(task, h): nn.Linear(cfg.d_model, 1)
                for task in cfg.tasks
                for h in cfg.horizons
            }
        )

    @staticmethod
    def _head_key(task: str, horizon: int) -> str:
        return f"{task}__h{horizon}"

    def forward(
        self, x: torch.Tensor
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        dict  {(task, horizon): (batch,)}
        """
        x = self.input_norm(x)
        _, (h_n, _) = self.lstm(x)
        enc = self.lstm_dropout(h_n[-1])  # last layer hidden state: (batch, hidden_dim)
        shared = self.shared_proj(enc)     # (batch, d_model)

        out: Dict[Tuple[str, int], torch.Tensor] = {}
        for task in self.cfg.tasks:
            for h in self.cfg.horizons:
                out[(task, h)] = self.heads[self._head_key(task, h)](shared).squeeze(-1)

        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultiTaskTrainer:
    """
    Fit MultiTaskMultiHorizonModel with early stopping.

    Handles NaN targets: samples where the target is NaN are masked out
    of the loss for that (task, horizon) head, so short sequences at the
    end of the dataset do not corrupt training.
    """

    def __init__(self, cfg: MultiTaskConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model: Optional[MultiTaskMultiHorizonModel] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _masked_mse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """MSE ignoring NaN targets."""
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return ((pred[mask] - target[mask]) ** 2).mean()

    def fit(
        self,
        train_dataset: MultiTaskDataset,
        val_dataset: MultiTaskDataset,
        verbose: bool = True,
    ) -> "MultiTaskTrainer":
        cfg = self.cfg

        self.model = MultiTaskMultiHorizonModel(cfg).to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=MultiTaskDataset.collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 4,
            shuffle=False,
            collate_fn=MultiTaskDataset.collate_fn,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # OneCycleLR: warmup (10%) then cosine decay to lr/1e4.
        # Steps per batch — gives fine-grained LR control and is significantly
        # better than ReduceLROnPlateau for longer training runs.
        steps_per_epoch = max(len(train_loader), 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.n_epochs,
            pct_start=0.10,       # 10% warmup
            div_factor=10.0,      # start at lr/10
            final_div_factor=1e4, # end at start_lr/1e4
            anneal_strategy="cos",
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0

        epoch_iter = range(cfg.n_epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="ep")

        for epoch in epoch_iter:
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = {k: v.to(self.device) for k, v in y_batch.items()}

                optimizer.zero_grad()
                preds = self.model(X_batch)

                loss = sum(
                    self._masked_mse(preds[k], y_batch[k]) for k in preds
                ) / max(len(preds), 1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()  # OneCycleLR steps per batch

                train_loss += loss.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
                    preds = self.model(X_batch)
                    loss = sum(
                        self._masked_mse(preds[k], y_batch[k]) for k in preds
                    ) / max(len(preds), 1)
                    val_loss += loss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            self.train_losses.append(avg_train)
            self.val_losses.append(avg_val)

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    train=f"{avg_train:.5f}",
                    val=f"{avg_val:.5f}",
                    lr=f"{current_lr:.2e}",
                )

            # --- Early stopping (only after min_epochs) ---
            if avg_val < best_val_loss - cfg.min_delta:
                best_val_loss = avg_val
                best_epoch = epoch + 1
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if epoch >= cfg.min_epochs and no_improve >= cfg.patience:
                    if verbose:
                        print(
                            f"\nEarly stopping at epoch {epoch + 1} "
                            f"(best val={best_val_loss:.6f} at epoch {best_epoch})"
                        )
                    break

        if verbose:
            print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> Dict[Tuple[str, int], np.ndarray]:
        """
        Deterministic predictions (dropout off).

        Parameters
        ----------
        X : (T, n_features) — full feature matrix

        Returns
        -------
        dict {(task, horizon): (T - seq_len + 1,) float32 array}
        The i-th prediction corresponds to index seq_len + i in X.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = self.cfg
        self.model.eval()

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        T = len(X)
        results: Dict[Tuple[str, int], List[float]] = {k: [] for k in [
            (t, h) for t in cfg.tasks for h in cfg.horizons
        ]}

        with torch.no_grad():
            for t in range(cfg.seq_len, T):
                x_seq = X_t[t - cfg.seq_len : t].unsqueeze(0)  # (1, seq, feat)
                out = self.model(x_seq)
                for k, v in out.items():
                    results[k].append(v.item())

        return {k: np.array(v, dtype=np.float32) for k, v in results.items()}

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_mc_samples: int = 30,
    ) -> Tuple[
        Dict[Tuple[str, int], np.ndarray],
        Dict[Tuple[str, int], np.ndarray],
    ]:
        """
        MC-Dropout uncertainty estimates.

        Returns (mean_predictions, std_predictions) with the same shape as
        predict().

        MC-Dropout activates dropout at inference time and averages over
        n_mc_samples stochastic forward passes.  The std across passes is a
        proxy for epistemic uncertainty.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")

        cfg = self.cfg
        self.model.train()  # Dropout active

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        T = len(X)
        n_steps = T - cfg.seq_len

        all_samples: Dict[Tuple[str, int], np.ndarray] = {
            (t, h): np.zeros((n_mc_samples, n_steps), dtype=np.float32)
            for t in cfg.tasks
            for h in cfg.horizons
        }

        with torch.no_grad():
            for s in range(n_mc_samples):
                for i, t in enumerate(range(cfg.seq_len, T)):
                    x_seq = X_t[t - cfg.seq_len : t].unsqueeze(0)
                    out = self.model(x_seq)
                    for k, v in out.items():
                        all_samples[k][s, i] = v.item()

        self.model.eval()

        means = {k: v.mean(axis=0) for k, v in all_samples.items()}
        stds = {k: v.std(axis=0) for k, v in all_samples.items()}
        return means, stds
