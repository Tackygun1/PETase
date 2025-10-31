"""I/O helpers for the project.

Lightweight functions for reading embeddings and label files. Kept small so
they're safe to import in CI and simple scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_embeddings(emb_path: Path) -> Dict[str, np.ndarray]:
    """Load a .npz embeddings file and return a mapping id -> vector.

    Parameters
    ----------
    emb_path:
            Path to a .npz file where each array key is an identifier and the
            corresponding value is a vector (numpy array).
    """
    emb = np.load(emb_path, allow_pickle=False)
    return {k: emb[k] for k in emb.files}


def load_labels_csv(labels_path: Path, id_col: str = "id", y_col: str = "label") -> pd.DataFrame:
    """Read a labels CSV and ensure required columns exist.

    Returns a dataframe with only (id_col, y_col) columns.
    """
    df = pd.read_csv(labels_path)
    missing = [c for c in (id_col, y_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file {labels_path} missing columns: {missing}")
    return df[[id_col, y_col]].copy()
