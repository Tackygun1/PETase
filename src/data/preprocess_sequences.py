"""
Sequence cleaning utilities for PETase data ingestion.
"""

from __future__ import annotations

import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequences(df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
    """
    Remove non-amino-acid characters and uppercase sequences.
    """
    out = df.copy()
    out[seq_col] = (
        out[seq_col]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
        .apply(lambda s: "".join(ch for ch in s if ch in VALID_AA))
    )
    return out


__all__ = ["clean_sequences"]
