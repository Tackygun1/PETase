"""
src/data/curate_training_set.py

Combines encoded sequences with experimental labels for model training.
"""

import pandas as pd
from typing import Dict
import numpy as np
from pathlib import Path


def curate_dataset(
    encoded_data: Dict[str, np.ndarray],
    labels_df: pd.DataFrame,
    out_path: Path = Path("data/processed/train_dataset.csv"),
):
    """
    Combine encoded features with labels into a single table.
    Expects labels_df to have columns ['id', 'label'] (e.g., stability, Tm).
    """
    merged = labels_df.copy()
    merged["feature_shape"] = merged["id"].map(
        lambda i: encoded_data[i].shape if i in encoded_data else None
    )
    merged = merged.dropna(subset=["feature_shape"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged
