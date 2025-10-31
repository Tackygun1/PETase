"""Model-specific utilities.

This module contains lightweight helpers used by model training code. General
I/O helpers (loading embeddings, reading labels) have been moved to
``src/utils/io.py``; the remaining function here focuses on aligning
embeddings with a labels dataframe to produce X/y for training.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def align_X_y(
    embeddings: Dict[str, np.ndarray],
    labels_df: pd.DataFrame,
    id_col: str = "id",
    y_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, Iterable[str]]:
    """
    Intersect IDs between embeddings and labels and return X, y in aligned order.
    Returns (X, y, kept_ids).

    The function preserves the order of IDs as found in ``labels_df`` and
    selects only those present in the ``embeddings`` mapping.
    """
    kept_ids = [rid for rid in labels_df[id_col].tolist() if rid in embeddings]
    if not kept_ids:
        raise ValueError("No overlapping IDs between embeddings and labels.")
    X = np.vstack([embeddings[rid] for rid in kept_ids])
    y = labels_df.set_index(id_col).loc[kept_ids, y_col].to_numpy()
    return X, y, kept_ids
