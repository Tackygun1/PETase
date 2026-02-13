#!/usr/bin/env python3
"""
Plot top-ranked final-shortlist candidates with multiple model outputs in one figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot top-ranked candidates with context/surrogate/Rosetta/AF2 values."
    )
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_with_af2_metrics.csv"),
        help="Merged final shortlist table.",
    )
    p.add_argument(
        "--surrogate-out",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_surrogate_predictions.csv"),
        help="Output CSV with surrogate predictions for shortlist IDs.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/candidates_100k_esm.npz"),
        help="Embeddings NPZ for candidate IDs.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("models/petase_pretrain_xgb_dtm.json"),
        help="Non-context surrogate XGBoost model.",
    )
    p.add_argument(
        "--calibration-metrics",
        type=Path,
        default=Path("results/surrogate_af2/calibration/metrics.csv"),
        help="Calibration metrics containing fit_slope and fit_intercept.",
    )
    p.add_argument(
        "--wt-centering",
        type=Path,
        default=Path("results/surrogate_af2/calibration/wt_centering.csv"),
        help="WT calibration table containing wt_pred_calibrated.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top N candidates by context residual to plot.",
    )
    p.add_argument(
        "--wt-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/wt_multimodel_evaluation.csv"),
        help="Optional WT multimodel evaluation row for reference overlay.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/figures_final_shortlist"),
        help="Output directory for the figure and top-table CSV.",
    )
    return p.parse_args()


def load_calibration_params(path: Path) -> Tuple[float, float]:
    slope = None
    intercept = None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            metric = (row.get("metric") or "").strip()
            val = (row.get("value") or "").strip()
            if not metric or not val:
                continue
            if metric == "fit_slope":
                slope = float(val)
            elif metric == "fit_intercept":
                intercept = float(val)
    if slope is None or intercept is None:
        raise SystemExit(f"Could not read fit_slope/fit_intercept from {path}")
    return slope, intercept


def load_wt_pred_cal(path: Path) -> float:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            val = (row.get("wt_pred_calibrated") or "").strip()
            if val:
                return float(val)
    raise SystemExit(f"Could not read wt_pred_calibrated from {path}")


def build_surrogate_predictions(
    ids: List[str],
    embeddings_path: Path,
    model_path: Path,
    calib_metrics: Path,
    wt_centering: Path,
    out_csv: Path,
) -> pd.DataFrame:
    slope, intercept = load_calibration_params(calib_metrics)
    wt_pred_cal = load_wt_pred_cal(wt_centering)

    emb_data = np.load(embeddings_path, allow_pickle=False)
    missing = [rid for rid in ids if rid not in emb_data.files]
    if missing:
        raise SystemExit(f"Missing {len(missing)} IDs in embeddings; first missing: {missing[0]}")

    X = np.vstack([emb_data[rid] for rid in ids])

    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    raw = model.predict(X)
    calibrated = (raw - intercept) / slope
    wt_centered = calibrated - wt_pred_cal

    df = pd.DataFrame(
        {
            "id": ids,
            "surrogate_pred_raw": raw.astype(float),
            "surrogate_pred_calibrated": calibrated.astype(float),
            "surrogate_dtm_wt_centered_cal": wt_centered.astype(float),
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def zscore_col(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def zscore_from_stats(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    if sd == 0.0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    shortlist = pd.read_csv(args.merged_csv)
    shortlist_ids = shortlist["id"].astype(str).tolist()

    surrogate = build_surrogate_predictions(
        ids=shortlist_ids,
        embeddings_path=args.embeddings,
        model_path=args.model,
        calib_metrics=args.calibration_metrics,
        wt_centering=args.wt_centering,
        out_csv=args.surrogate_out,
    )

    df = shortlist.merge(surrogate, on="id", how="left")
    df = df.sort_values("pred_residual", ascending=False).head(args.top_n).copy()
    df = df.reset_index(drop=True)

    # Raw metrics table for downstream slide use.
    raw_cols = [
        "id",
        "pred_residual",
        "pred_abs",
        "surrogate_dtm_wt_centered_cal",
        "ddg_reu",
        "af2_ranked0_plddt_mean",
        "af2_ranked0_plddt_key_mean",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = args.out_dir / "top_ranked_multimodel_values.csv"
    df[raw_cols].to_csv(raw_out, index=False)

    wt_row = None
    if args.wt_csv.exists():
        wt_df = pd.read_csv(args.wt_csv)
        if len(wt_df):
            wt_series = wt_df.iloc[0]
            wt_row = {
                "id": str(wt_series.get("id", "IsPETaseWT")) + " (WT ref)",
                "pred_residual": float(pd.to_numeric(pd.Series([wt_series.get("pred_residual", np.nan)]), errors="coerce").iloc[0]),
                "pred_abs": float(pd.to_numeric(pd.Series([wt_series.get("pred_abs", np.nan)]), errors="coerce").iloc[0]),
                "surrogate_dtm_wt_centered_cal": float(
                    pd.to_numeric(pd.Series([wt_series.get("surrogate_dtm_wt_centered_cal", np.nan)]), errors="coerce").iloc[0]
                ),
                "ddg_reu": float(pd.to_numeric(pd.Series([wt_series.get("ddg_reu_reference", np.nan)]), errors="coerce").iloc[0]),
                "af2_ranked0_plddt_mean": float(
                    pd.to_numeric(pd.Series([wt_series.get("af2_ranked0_plddt_mean", np.nan)]), errors="coerce").iloc[0]
                ),
                "af2_ranked0_plddt_key_mean": float(
                    pd.to_numeric(pd.Series([wt_series.get("af2_ranked0_plddt_key_mean", np.nan)]), errors="coerce").iloc[0]
                ),
            }

    # Matrix for heatmap (z-scored per metric for comparability).
    metric_cols = [
        "pred_residual",
        "pred_abs",
        "surrogate_dtm_wt_centered_cal",
        "ddg_reu",
        "af2_ranked0_plddt_mean",
        "af2_ranked0_plddt_key_mean",
    ]
    labels = [
        "Context residual dTm (C)",
        "Context absolute dTm (C)",
        "Surrogate dTm wt-centered (C)",
        "Rosetta ddG (REU)",
        "AF2 pLDDT mean",
        "AF2 key-site pLDDT",
    ]

    plot_df = df.copy()
    if wt_row is not None:
        plot_df = pd.concat([plot_df, pd.DataFrame([wt_row])], ignore_index=True)

    raw_candidates = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in metric_cols])
    raw_plot = np.column_stack([pd.to_numeric(plot_df[c], errors="coerce").to_numpy(dtype=float) for c in metric_cols])
    zcols = []
    for i in range(raw_candidates.shape[1]):
        cand = raw_candidates[:, i]
        mu = float(np.nanmean(cand))
        sd = float(np.nanstd(cand, ddof=1))
        zcols.append(zscore_from_stats(raw_plot[:, i], mu, sd))
    zmat = np.column_stack(zcols)

    # For display labels, keep raw values with compact formatting.
    text_vals = raw_plot.copy()

    fig_h = max(8.0, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    im = ax.imshow(zmat, aspect="auto", cmap="coolwarm", vmin=-2.3, vmax=2.3)

    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["id"].tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title(
        f"Top {len(df)} Final Candidates + WT Reference: Context, Surrogate, Rosetta, and AF2 Metrics\n"
        "(Cell color: z-score vs top-candidate distribution; text: raw value)",
        fontsize=14,
        pad=14,
    )

    for i in range(text_vals.shape[0]):
        for j in range(text_vals.shape[1]):
            v = text_vals[i, j]
            if j == 3:
                s = f"{v:.1f}"
            elif j >= 4:
                s = f"{v:.2f}"
            else:
                s = f"{v:.3f}"
            ax.text(j, i, s, ha="center", va="center", fontsize=7, color="black")

    if wt_row is not None:
        wt_idx = len(plot_df) - 1
        ax.axhline(wt_idx - 0.5, color="black", linewidth=1.6)
        ax.axhline(wt_idx + 0.5, color="black", linewidth=1.0)
        ax.text(
            -0.45,
            wt_idx,
            "WT",
            ha="right",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
            transform=ax.transData,
            clip_on=False,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Within-metric z-score")
    save_figure(fig, args.out_dir / "fig9_top_ranked_multimodel_values")

    # Also write a compact summary text CSV for abstract-style numbers.
    summary_rows = []
    for col in raw_cols[1:]:
        s = pd.to_numeric(df[col], errors="coerce")
        summary_rows.append((col, float(s.mean()), float(s.min()), float(s.max())))
    with open(args.out_dir / "summary_top_ranked_multimodel_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "min", "max"])
        w.writerows(summary_rows)

    print(f"Wrote surrogate predictions: {args.surrogate_out}")
    print(f"Wrote top raw values table: {raw_out}")
    print(f"Wrote figure: {args.out_dir / 'fig9_top_ranked_multimodel_values.png'}")


if __name__ == "__main__":
    main()
