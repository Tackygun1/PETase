"""
Uncertainty utilities for deep-ensemble surrogate models.

Includes:
- Ensemble mean/variance computation with optional variance scaling.
- Simple variance calibration (fit a scalar to align predicted variance with residuals).
- Reliability statistics used in calibration reporting.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def ensemble_mean_var(
    preds: torch.Tensor, scale: torch.Tensor | float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predictive mean/variance from ensemble outputs.

    Args:
        preds: Tensor of shape (E, N, T) for E ensemble members, N samples, T targets.
        scale: Scalar or vector to scale variance (useful for calibration).
    """
    mean = preds.mean(dim=0)
    var_model = preds.var(dim=0, unbiased=False)
    if isinstance(scale, torch.Tensor):
        var = var_model * scale.view(1, -1)
    else:
        var = var_model * scale
    return mean, var


def fit_variance_scaler(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Fit a per-target scalar that aligns predicted variance to squared error.
    """
    resid2 = (y_true - mean) ** 2
    # avoid divide-by-zero
    var = np.clip(var, 1e-8, None)
    scale = (resid2 / var).mean(axis=0)
    return np.maximum(scale, 1e-3)


def reliability_stats(
    y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute coarse calibration statistics for regression (expected vs. observed error).
    """
    se = (y_true - mean) ** 2
    preds = var
    ratios = se / np.clip(preds, 1e-8, None)
    calib_err = float(np.abs(ratios.mean() - 1.0))
    sharpness = float(preds.mean())
    return {"calib_error": calib_err, "sharpness": sharpness}
