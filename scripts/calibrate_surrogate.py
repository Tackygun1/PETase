#!/usr/bin/env python3
"""
Calibrate surrogate predictions against labeled data.
Produces metrics + plots (scatter, residuals, calibration curve).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LABEL_COLS = ("stability", "DTM_C", "DTM", "TM_C", "TM", "DDG")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate surrogate predictions vs labels.")
    p.add_argument(
        "--model",
        type=Path,
        default=Path("models/petase_pretrain_xgb_dtm.json"),
        help="XGBoost model JSON.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/petase_pretrain_embeddings.npz"),
        help="Embeddings NPZ keyed by id.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/petase_pretrain_labels.csv"),
        help="Labels CSV with id and target column.",
    )
    p.add_argument(
        "--label-col",
        type=str,
        default="",
        help="Label column name (auto-detect if omitted).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/calibration"),
        help="Output directory for metrics + plots.",
    )
    p.add_argument(
        "--apply-to",
        type=Path,
        default=None,
        help="Optional predictions CSV (id,prediction) to calibrate.",
    )
    p.add_argument(
        "--apply-out",
        type=Path,
        default=Path("results/surrogate_af2/predictions_calibrated.csv"),
        help="Output path for calibrated predictions CSV.",
    )
    p.add_argument(
        "--apply-wt-center",
        action="store_true",
        help="Also write WT-centered predictions (raw and calibrated).",
    )
    p.add_argument(
        "--wt-id",
        type=str,
        default="IsPETaseWT",
        help="WT sequence ID used for centering.",
    )
    p.add_argument(
        "--wt-embeddings",
        type=Path,
        default=Path("data/processed/petase_esm_embeddings.npz"),
        help="Embeddings NPZ containing the WT id (used for centering).",
    )
    p.add_argument("--n-bins", type=int, default=10, help="Bins for calibration curve.")
    return p.parse_args()


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def infer_label_col(fieldnames: List[str], requested: str) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Label column {requested} not found in {fieldnames}")
        return requested
    for col in DEFAULT_LABEL_COLS:
        if col in fieldnames:
            return col
    raise ValueError(f"Could not infer label column from {fieldnames}")


def load_labels(path: Path, label_col: str) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        col = infer_label_col(reader.fieldnames, label_col)
        for row in reader:
            rid = (row.get("id") or "").strip()
            val = (row.get(col) or "").strip()
            if not rid or not val:
                continue
            try:
                labels[rid] = float(val)
            except ValueError:
                continue
    return labels


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mean = float(y_true.mean())
    ss_tot = float(((y_true - mean) ** 2).sum())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    spear = spearman(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spear,
        "mean_true": mean,
        "mean_pred": float(y_pred.mean()),
        "n": int(len(y_true)),
    }


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> Tuple[float, float]:
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=25, color="#3b6fb6", alpha=0.8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", label="Ideal")
    plt.plot(lims, [slope * x + intercept for x in lims], color="#cf4d3b", label="Fit")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    return float(slope), float(intercept)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    resid = y_pred - y_true
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, resid, s=25, color="#6d9f71", alpha=0.8)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (pred - true)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_resid_hist(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    resid = y_pred - y_true
    plt.figure(figsize=(5, 4))
    plt.hist(resid, bins=20, color="#8fb3d9", edgecolor="white")
    plt.axvline(0, color="#333333", linewidth=1)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int, out: Path) -> None:
    # Bin by predicted quantiles
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_pred, qs)
    exp = []
    obs = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        if not mask.any():
            continue
        exp.append(float(y_pred[mask].mean()))
        obs.append(float(y_true[mask].mean()))
    if not exp:
        return
    exp = np.array(exp)
    obs = np.array(obs)
    plt.figure(figsize=(5, 4))
    plt.plot(exp, obs, "o-", color="#3b6fb6", label="Observed")
    plt.plot([exp.min(), exp.max()], [exp.min(), exp.max()], "k--", label="Ideal")
    plt.xlabel("Mean predicted (bin)")
    plt.ylabel("Mean observed (bin)")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_distributions(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(y_true, bins=20, alpha=0.6, label="Observed", color="#3b6fb6")
    plt.hist(y_pred, bins=20, alpha=0.6, label="Predicted", color="#cf4d3b")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    embeddings = load_embeddings(args.embeddings)
    labels = load_labels(args.labels, args.label_col)
    ids = [rid for rid in labels.keys() if rid in embeddings]
    if not ids:
        raise SystemExit("No overlapping IDs between labels and embeddings.")

    X = np.vstack([embeddings[rid] for rid in ids])
    y_true = np.array([labels[rid] for rid in ids], dtype=float)

    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(args.model)
    y_pred = model.predict(X)

    stats = metrics(y_true, y_pred)
    slope, intercept = plot_scatter(y_true, y_pred, args.out_dir / "scatter_pred_vs_true.png")
    plot_residuals(y_true, y_pred, args.out_dir / "residuals_vs_pred.png")
    plot_resid_hist(y_true, y_pred, args.out_dir / "residuals_hist.png")
    plot_calibration(y_true, y_pred, args.n_bins, args.out_dir / "calibration_curve.png")
    plot_distributions(y_true, y_pred, args.out_dir / "distributions.png")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stats.items():
            writer.writerow([k, f"{v:.6f}"])
        writer.writerow(["fit_slope", f"{slope:.6f}"])
        writer.writerow(["fit_intercept", f"{intercept:.6f}"])

    with open(args.out_dir / "predictions_vs_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "observed", "predicted", "residual"])
        for rid, yt, yp in zip(ids, y_true, y_pred):
            writer.writerow([rid, f"{yt:.6f}", f"{yp:.6f}", f"{(yp - yt):.6f}"])

    if args.apply_to:
        if slope == 0:
            raise SystemExit("Calibration slope is zero; cannot apply.")
        wt_pred_raw = None
        wt_pred_cal = None
        if args.apply_wt_center:
            emb_paths = [args.wt_embeddings, args.embeddings]
            wt_emb = None
            for p in emb_paths:
                if not p or not p.exists():
                    continue
                data = np.load(p, allow_pickle=False)
                if args.wt_id in data.files:
                    wt_emb = data[args.wt_id]
                    break
            if wt_emb is None:
                raise SystemExit(
                    f"WT id {args.wt_id} not found in embeddings; check --wt-embeddings."
                )
            wt_pred_raw = float(model.predict(wt_emb.reshape(1, -1))[0])
            wt_pred_cal = (wt_pred_raw - intercept) / slope

        with open(args.apply_to, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames or "prediction" not in reader.fieldnames:
                raise SystemExit(f"Missing id/prediction columns in {args.apply_to}")
            rows = []
            for row in reader:
                rid = (row.get("id") or "").strip()
                val = (row.get("prediction") or "").strip()
                if not rid or not val:
                    continue
                try:
                    pred = float(val)
                except ValueError:
                    continue
                # Invert linear fit: pred = slope * true + intercept
                calibrated = (pred - intercept) / slope
                raw_center = None
                cal_center = None
                if args.apply_wt_center and wt_pred_raw is not None and wt_pred_cal is not None:
                    raw_center = pred - wt_pred_raw
                    cal_center = calibrated - wt_pred_cal
                rows.append((rid, pred, calibrated, raw_center, cal_center))

        args.apply_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.apply_out, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["id", "prediction", "prediction_calibrated"]
            if args.apply_wt_center:
                header.extend(["prediction_wt_centered", "prediction_calibrated_wt_centered"])
            writer.writerow(header)
            for rid, pred, calibrated, raw_center, cal_center in rows:
                row = [rid, f"{pred:.6f}", f"{calibrated:.6f}"]
                if args.apply_wt_center:
                    row.extend(
                        [
                            f"{raw_center:.6f}" if raw_center is not None else "",
                            f"{cal_center:.6f}" if cal_center is not None else "",
                        ]
                    )
                writer.writerow(row)
        print(f"Wrote calibrated predictions to {args.apply_out}")
        if args.apply_wt_center and wt_pred_raw is not None:
            with open(args.out_dir / "wt_centering.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["wt_id", "wt_pred_raw", "wt_pred_calibrated"])
                writer.writerow([args.wt_id, f"{wt_pred_raw:.6f}", f"{wt_pred_cal:.6f}"])

    print(f"Wrote calibration outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
