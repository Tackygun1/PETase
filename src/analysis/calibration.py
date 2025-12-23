"""
Calibration utilities for surrogate ensembles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def reliability_curve(
    y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    se = (y_true - mean) ** 2
    expected = var
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(expected, quantiles)
    obs = []
    exp = []
    for i in range(n_bins):
        mask = (expected >= bins[i]) & (expected <= bins[i + 1])
        if not mask.any():
            continue
        obs.append(se[mask].mean())
        exp.append(expected[mask].mean())
    return np.array(exp), np.array(obs)


def plot_reliability(exp: np.ndarray, obs: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(exp, obs, "o-", label="Observed")
    plt.plot([exp.min(), exp.max()], [exp.min(), exp.max()], "k--", label="Ideal")
    plt.xlabel("Predicted variance")
    plt.ylabel("Observed squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


__all__ = ["reliability_curve", "plot_reliability"]
