"""
Deep-ensemble surrogate model for PETase variant scoring.

Implements:
- YAML-driven configuration (data + model + training + output).
- MLP regressors trained as an ensemble for calibrated mean/variance.
- Automatic alignment of embeddings and labels (via utils.io / models.utils).
- Save/load helpers that persist config, metrics, and state dicts.

The surrogate predicts one or more regression targets (e.g., stability, activity).
Uncertainty comes from ensemble dispersion plus a learned variance scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split

from ..utils.io import load_embeddings, load_labels_csv
from .utils import align_X_y
from .uncertainty import (
    ensemble_mean_var,
    fit_variance_scaler,
    reliability_stats,
)


# --------------------------
# Configuration
# --------------------------
@dataclass
class SurrogateConfig:
    # Data
    embeddings_path: str = "data/processed/esm_embeddings.npz"
    labels_path: str = "data/processed/labels.csv"
    id_col: str = "id"
    target_cols: Tuple[str, ...] = ("stability", "activity")
    dropna: bool = True
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True

    # Model
    hidden_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.1
    ensemble_size: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 128
    patience: int = 8  # early stopping
    min_delta: float = 1e-4

    # Output
    output_dir: str = "models"
    model_prefix: str = "surrogate"


def load_config(path: Path | str) -> SurrogateConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    def get(section: str, key: str, default):
        return raw.get(section, {}).get(key, raw.get(key, default))

    targets = get("data", "target_cols", SurrogateConfig.target_cols)
    if isinstance(targets, str):
        target_cols = (targets,)
    else:
        target_cols = tuple(targets)

    hidden = get("model", "hidden_dims", SurrogateConfig.hidden_dims)
    hidden_dims = tuple(int(h) for h in hidden)

    return SurrogateConfig(
        embeddings_path=get("data", "embeddings_path", SurrogateConfig.embeddings_path),
        labels_path=get("data", "labels_path", SurrogateConfig.labels_path),
        id_col=get("data", "id_col", SurrogateConfig.id_col),
        target_cols=target_cols,
        dropna=bool(get("data", "dropna", SurrogateConfig.dropna)),
        test_size=float(get("training", "test_size", SurrogateConfig.test_size)),
        random_state=int(get("training", "random_state", SurrogateConfig.random_state)),
        shuffle=bool(get("training", "shuffle", SurrogateConfig.shuffle)),
        hidden_dims=hidden_dims,
        dropout=float(get("model", "dropout", SurrogateConfig.dropout)),
        ensemble_size=int(get("model", "ensemble_size", SurrogateConfig.ensemble_size)),
        lr=float(get("training", "lr", SurrogateConfig.lr)),
        weight_decay=float(get("training", "weight_decay", SurrogateConfig.weight_decay)),
        epochs=int(get("training", "epochs", SurrogateConfig.epochs)),
        batch_size=int(get("training", "batch_size", SurrogateConfig.batch_size)),
        patience=int(get("training", "patience", SurrogateConfig.patience)),
        min_delta=float(get("training", "min_delta", SurrogateConfig.min_delta)),
        output_dir=get("output", "output_dir", SurrogateConfig.output_dir),
        model_prefix=get("output", "model_prefix", SurrogateConfig.model_prefix),
    )


# --------------------------
# Model definitions
# --------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int], dropout: float):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurrogateEnsemble:
    """
    Deep ensemble that returns predictive mean/variance from member outputs.
    """

    def __init__(self, cfg: SurrogateConfig, input_dim: int, n_targets: int):
        self.cfg = cfg
        self.n_targets = n_targets
        self.models = nn.ModuleList(
            [
                MLPRegressor(
                    input_dim=input_dim,
                    output_dim=n_targets,
                    hidden_dims=cfg.hidden_dims,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.ensemble_size)
            ]
        )
        self.optims = [
            torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for m in self.models
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models.to(self.device)
        self.var_scale = torch.ones(n_targets, device=self.device)

    # -------- Data handling --------
    @staticmethod
    def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).float()
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    # -------- Training --------
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        train_loader = self._make_loader(X, y, batch_size=self.cfg.batch_size, shuffle=True)
        best_val = np.inf
        patience_counter = 0

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                batch_loss = 0.0
                for model, opt in zip(self.models, self.optims):
                    opt.zero_grad()
                    preds = model(xb)
                    loss = F.mse_loss(preds, yb)
                    loss.backward()
                    opt.step()
                    batch_loss += loss.item()
                epoch_loss += batch_loss / len(self.models)

            avg_loss = epoch_loss / len(train_loader)
            if avg_loss + self.cfg.min_delta < best_val:
                best_val = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                break

        return {"loss": float(best_val)}

    # -------- Prediction --------
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.models.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).float().to(self.device)
            preds = torch.stack([m(xb) for m in self.models], dim=0)  # (E, N, T)
            mean, var = ensemble_mean_var(preds, scale=self.var_scale)
        return mean.cpu().numpy(), var.cpu().numpy()

    def calibrate_variance(self, X_val: np.ndarray, y_val: np.ndarray):
        mean, var = self.predict(X_val)
        scale = fit_variance_scaler(y_val, mean, var)
        self.var_scale = torch.tensor(scale, device=self.device)
        return scale

    # -------- Persistence --------
    def save(self, out_dir: Path, cfg: SurrogateConfig, metrics: Dict[str, float]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg": asdict(cfg),
            "metrics": metrics,
            "state_dicts": [m.state_dict() for m in self.models],
            "var_scale": self.var_scale.cpu().tolist(),
        }
        out_path = out_dir / f"{cfg.model_prefix}.pt"
        torch.save(state, out_path)
        # also persist JSON metadata for quick reads
        meta = out_dir / f"{cfg.model_prefix}_meta.json"
        with open(meta, "w") as f:
            json.dump({"metrics": metrics, "config": asdict(cfg)}, f, indent=2)
        return out_path

    @classmethod
    def load(cls, model_path: Path | str) -> "SurrogateEnsemble":
        model_path = Path(model_path)
        state = torch.load(model_path, map_location="cpu")
        cfg = SurrogateConfig(**state["cfg"])
        dummy_X = np.zeros((1, state["state_dicts"][0]["net.0.weight"].shape[1]), dtype=np.float32)
        inst = cls(cfg, input_dim=dummy_X.shape[1], n_targets=len(cfg.target_cols))
        for model, sd in zip(inst.models, state["state_dicts"]):
            model.load_state_dict(sd)
        inst.var_scale = torch.tensor(state.get("var_scale", [1.0] * len(cfg.target_cols)))
        return inst


# --------------------------
# High-level training helper
# --------------------------
def train_from_config(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    emb = load_embeddings(Path(cfg.embeddings_path))
    labels_df = load_labels_csv(Path(cfg.labels_path), id_col=cfg.id_col, y_col=cfg.target_cols)

    # keep only target columns; allow dropna if requested
    missing_targets = [t for t in cfg.target_cols if t not in labels_df.columns]
    if missing_targets:
        raise ValueError(f"Targets missing in labels file: {missing_targets}")
    if cfg.dropna:
        labels_df = labels_df.dropna(subset=cfg.target_cols)

    X, y, kept = align_X_y(
        emb,
        labels_df[[cfg.id_col, *cfg.target_cols]],
        id_col=cfg.id_col,
        y_col=cfg.target_cols,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=cfg.shuffle,
    )

    ensemble = SurrogateEnsemble(cfg, input_dim=X.shape[1], n_targets=y.shape[1])
    train_metrics = ensemble.fit(X_tr, y_tr)
    scale = ensemble.calibrate_variance(X_te, y_te)
    pred_mean, pred_var = ensemble.predict(X_te)
    calib = reliability_stats(y_te, pred_mean, pred_var)

    metrics = {
        **train_metrics,
        "var_scale": scale.tolist() if isinstance(scale, np.ndarray) else list(scale),
        **{f"calib_{k}": v for k, v in calib.items()},
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "embedding_dim": int(X.shape[1]),
    }

    out_dir = Path(cfg.output_dir)
    ensemble.save(out_dir, cfg, metrics)

    # persist kept IDs for downstream analysis
    joblib.dump({"kept_ids": kept}, out_dir / f"{cfg.model_prefix}_kept_ids.pkl")
    return metrics


__all__ = ["SurrogateConfig", "load_config", "SurrogateEnsemble", "train_from_config"]
