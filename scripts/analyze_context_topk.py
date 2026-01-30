#!/usr/bin/env python3
"""
Analyze top-k context surrogate predictions vs background with stats + plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze top-k context predictions.")
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/context_candidate_predictions.csv"),
        help="CSV with prediction scores.",
    )
    p.add_argument(
        "--score-col",
        default="pred_residual",
        help="Column name for the prediction score.",
    )
    p.add_argument("--topk", type=int, default=20, help="Top-k cutoff.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/context_topk_eval"),
        help="Output directory for plots and stats.",
    )
    return p.parse_args()


def _percentile_ranks(values: np.ndarray, all_values: np.ndarray) -> np.ndarray:
    return stats.rankdata(all_values, method="average")[np.searchsorted(
        np.sort(all_values), values
    )] / len(all_values) * 100.0


def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "min": float(np.min(x)) if len(x) else float("nan"),
        "max": float(np.max(x)) if len(x) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    if args.score_col not in df.columns:
        raise SystemExit(f"Missing score column {args.score_col} in {args.predictions}.")

    df = df.dropna(subset=[args.score_col]).copy()
    if df.empty:
        raise SystemExit("No prediction rows found.")

    df = df.sort_values(args.score_col, ascending=False).reset_index(drop=True)
    topk = df.head(args.topk).copy()
    background = df.iloc[args.topk :].copy()

    all_vals = df[args.score_col].to_numpy()
    top_vals = topk[args.score_col].to_numpy()
    bg_vals = background[args.score_col].to_numpy()

    topk["percentile"] = [
        stats.percentileofscore(all_vals, v, kind="rank") for v in top_vals
    ]

    # Summary stats
    summary = {
        "n_all": len(all_vals),
        "n_topk": len(top_vals),
        "n_background": len(bg_vals),
        "topk_mean": _safe_stats(top_vals)["mean"],
        "topk_median": _safe_stats(top_vals)["median"],
        "topk_std": _safe_stats(top_vals)["std"],
        "bg_mean": _safe_stats(bg_vals)["mean"],
        "bg_median": _safe_stats(bg_vals)["median"],
        "bg_std": _safe_stats(bg_vals)["std"],
        "all_mean": _safe_stats(all_vals)["mean"],
        "all_median": _safe_stats(all_vals)["median"],
        "all_std": _safe_stats(all_vals)["std"],
        "topk_mean_percentile": float(np.mean(topk["percentile"])),
        "topk_min_percentile": float(np.min(topk["percentile"])),
        "topk_max_percentile": float(np.max(topk["percentile"])),
    }

    # Effect size (Cohen's d) vs background
    if len(bg_vals) > 1 and len(top_vals) > 1:
        pooled = np.sqrt(((len(top_vals) - 1) * np.var(top_vals, ddof=1) +
                          (len(bg_vals) - 1) * np.var(bg_vals, ddof=1)) /
                         (len(top_vals) + len(bg_vals) - 2))
        summary["cohens_d"] = float((np.mean(top_vals) - np.mean(bg_vals)) / pooled) if pooled else float("nan")
    else:
        summary["cohens_d"] = float("nan")

    # Statistical tests vs background
    if len(bg_vals) > 0:
        mw = stats.mannwhitneyu(top_vals, bg_vals, alternative="greater")
        ks = stats.ks_2samp(top_vals, bg_vals)
        ttest = stats.ttest_ind(top_vals, bg_vals, equal_var=False, alternative="greater")
        summary["mannwhitney_u"] = float(mw.statistic)
        summary["mannwhitney_p"] = float(mw.pvalue)
        summary["ks_stat"] = float(ks.statistic)
        summary["ks_p"] = float(ks.pvalue)
        summary["ttest_stat"] = float(ttest.statistic)
        summary["ttest_p"] = float(ttest.pvalue)

    # Percentile thresholds
    for p in (50, 75, 90, 95, 99):
        summary[f"p{p}"] = float(np.percentile(all_vals, p))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    topk.to_csv(args.out_dir / "topk_with_percentiles.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(all_vals, bins=40, alpha=0.7, label="all")
    plt.hist(top_vals, bins=20, alpha=0.7, label="topk")
    plt.xlabel(args.score_col)
    plt.ylabel("count")
    plt.title("Top-k vs All Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "hist_topk_vs_all.png", dpi=180)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot([bg_vals, top_vals], labels=["background", "topk"])
    plt.ylabel(args.score_col)
    plt.title("Top-k vs Background")
    plt.tight_layout()
    plt.savefig(args.out_dir / "boxplot_topk_vs_bg.png", dpi=180)
    plt.close()

    # ECDF
    def ecdf(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x_all, y_all = ecdf(all_vals)
    x_top, y_top = ecdf(top_vals)
    plt.figure(figsize=(7, 5))
    plt.plot(x_all, y_all, label="all")
    plt.plot(x_top, y_top, label="topk")
    plt.xlabel(args.score_col)
    plt.ylabel("ECDF")
    plt.title("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "ecdf_topk_vs_all.png", dpi=180)
    plt.close()

    # Rank plot
    plt.figure(figsize=(8, 5))
    plt.plot(df[args.score_col].to_numpy(), label="all")
    plt.scatter(
        np.arange(len(top_vals)),
        top_vals,
        color="red",
        label="topk",
        s=12,
    )
    plt.xlabel("rank (sorted)")
    plt.ylabel(args.score_col)
    plt.title("Ranked scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "ranked_scores.png", dpi=180)
    plt.close()

    print(f"Wrote stats and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
