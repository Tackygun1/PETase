"""
Pareto frontier utilities for stability/activity trade-offs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pts = df[[x_col, y_col]].to_numpy()
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        if not is_pareto[i]:
            continue
        is_pareto[is_pareto] = np.any(pts[is_pareto] > p, axis=1) | np.all(
            pts[is_pareto] == p, axis=1
        )
        is_pareto[i] = True
    return df[is_pareto].copy()


__all__ = ["pareto_front"]
