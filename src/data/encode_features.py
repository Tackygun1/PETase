"""
src/data/encode_features.py

Encodes sequences into numerical representations for ML models.

Note: One-hot encoding should be switched to ESM embeddings in future iterations.
"""

import numpy as np
import pandas as pd
from typing import Dict

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def one_hot_encode(sequence: str) -> np.ndarray:
    """Convert a sequence string to a 2D one-hot numpy array."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    mat = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            mat[i, aa_to_idx[aa]] = 1.0
    return mat


def encode_dataframe(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return dict: id → encoded numpy array"""
    return {row["id"]: one_hot_encode(row["sequence"]) for _, row in df.iterrows()}
