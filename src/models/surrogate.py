"""
Configurable surrogate model for PETase property prediction from embeddings.

Features:
- YAML config support (paths + hyperparams)
- Train/validate split with metrics
- Save/load model + metadata (JSON)
- CLI entry points (safe to ignore on mac; no auto-run)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..utils.io import load_embeddings, load_labels_csv
from .utils import align_X_y


# --------------------------
# Configuration
# --------------------------
@dataclass
class SurrogateConfig:
    # Data
    embeddings_path: str = "data/processed/esm_embeddings.npz"
    labels_path: str = "data/processed/labels.csv"
    id_col: str = "id"
    y_col: str = "label"

    # Model
    model_type: str = "RandomForest"  # currently only RF; extend later if needed
    n_estimators: int = 200
    max_depth: Optional[int] = 10
    random_state: int = 42

    # Training
    test_size: float = 0.2
    shuffle: bool = True

    # Output
    output_dir: str = "models"
    model_filename: str = "surrogate.pkl"
    meta_filename: str = "surrogate_meta.json"


def load_config(path: Path | str) -> SurrogateConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Support nested layout {data:{}, surrogate:{}, output:{}}, but keep keys flexible
    # Flatten by reading known keys if present
    def get(section: str, key: str, default: Any):
        return raw.get(section, {}).get(key, raw.get(key, default))

    return SurrogateConfig(
        embeddings_path=get("data", "embeddings_path", SurrogateConfig.embeddings_path),
        labels_path=get("data", "labels_path", SurrogateConfig.labels_path),
        id_col=get("data", "id_col", SurrogateConfig.id_col),
        y_col=get("data", "y_col", SurrogateConfig.y_col),
        model_type=get("surrogate", "model_type", SurrogateConfig.model_type),
        n_estimators=int(get("surrogate", "n_estimators", SurrogateConfig.n_estimators)),
        max_depth=(
            None
            if (v := get("surrogate", "max_depth", SurrogateConfig.max_depth)) in (None, "null")
            else int(v)
        ),
        random_state=int(get("surrogate", "random_state", SurrogateConfig.random_state)),
        test_size=float(get("surrogate", "test_size", SurrogateConfig.test_size)),
        shuffle=bool(get("surrogate", "shuffle", SurrogateConfig.shuffle)),
        output_dir=get("output", "output_dir", SurrogateConfig.output_dir),
        model_filename=get("output", "model_filename", SurrogateConfig.model_filename),
        meta_filename=get("output", "meta_filename", SurrogateConfig.meta_filename),
    )


# --------------------------
# Model Wrapper
# --------------------------
class SurrogateModel:
    def __init__(self, cfg: SurrogateConfig):
        self.cfg = cfg
        self.model = self._build_model()
        self.metrics: Dict[str, float] = {}
        self._kept_ids: list[str] = []

    def _build_model(self):
        if self.cfg.model_type.lower() == "randomforest":
            return RandomForestRegressor(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                random_state=self.cfg.random_state,
                n_jobs=-1,  # train fast on Linux box
            )
        raise ValueError(f"Unsupported model_type: {self.cfg.model_type}")

    # -------- Data I/O --------
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        emb = load_embeddings(Path(self.cfg.embeddings_path))
        labels_df = load_labels_csv(Path(self.cfg.labels_path), self.cfg.id_col, self.cfg.y_col)
        X, y, kept = align_X_y(emb, labels_df, self.cfg.id_col, self.cfg.y_col)
        self._kept_ids = list(kept)
        return X, y

    # -------- Train/Eval --------
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            shuffle=self.cfg.shuffle,
        )
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        self.metrics = {
            "r2": float(r2_score(y_te, preds)),
            "mse": float(mean_squared_error(y_te, preds)),
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "model_type": self.cfg.model_type,
        }
        return self.metrics

    # -------- Predict --------
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        return self.model.predict(X_new)

    # -------- Save/Load --------
    def save(self) -> Path:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / self.cfg.model_filename
        meta_path = out_dir / self.cfg.meta_filename

        joblib.dump(self.model, model_path)
        meta = {
            "config": vars(self.cfg),
            "metrics": self.metrics,
            "kept_ids": self._kept_ids,  # IDs used during alignment
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return model_path

    @classmethod
    def load(cls, model_dir: Path | str) -> "SurrogateModel":
        """
        Load a trained model + metadata. Returns a SurrogateModel with cfg restored.
        """
        model_dir = Path(model_dir)
        # Infer meta + model paths by scanning common filenames
        meta_candidates = ["surrogate_meta.json", "meta.json"]
        model_candidates = ["surrogate.pkl", "model.pkl"]

        meta_path = next((model_dir / p for p in meta_candidates if (model_dir / p).exists()), None)
        model_path = next(
            (model_dir / p for p in model_candidates if (model_dir / p).exists()), None
        )

        if meta_path is None or model_path is None:
            raise FileNotFoundError(f"Could not find model/meta in {model_dir}")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        cfg = SurrogateConfig(**meta["config"])
        inst = cls(cfg)
        inst.model = joblib.load(model_path)
        inst.metrics = meta.get("metrics", {})
        inst._kept_ids = meta.get("kept_ids", [])
        return inst


# --------------------------
# CLI (safe, won’t auto-run)
# --------------------------
def cli_train_from_config(config_path: str) -> Dict[str, float]:
    """
    Programmatic entrypoint, e.g.:
        from src.models.surrogate import cli_train_from_config
        cli_train_from_config("config/experiment.yaml")
    """
    cfg = load_config(config_path)
    model = SurrogateModel(cfg)
    X, y = model.load_training_data()
    metrics = model.train(X, y)
    model.save()
    return metrics
