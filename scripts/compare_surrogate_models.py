#!/usr/bin/env python3
"""
Compare optimistic (baseline) vs pessimistic PETase-finetuned context surrogate predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare surrogate prediction distributions.")
    p.add_argument(
        "--old",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions.csv"),
        help="Old (optimistic) predictions CSV.",
    )
    p.add_argument(
        "--new",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv"),
        help="New (pessimistic/fine-tuned) predictions CSV.",
    )
    p.add_argument("--score-col", default="pred_residual", help="Prediction column name.")
    p.add_argument("--topk", type=int, default=20, help="Top-k for overlap/summary.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/compare_models"),
        help="Output directory.",
    )
    return p.parse_args()


def ecdf(vals: np.ndarray):
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    args = parse_args()
    old = pd.read_csv(args.old)
    new = pd.read_csv(args.new)

    if args.score_col not in old.columns or args.score_col not in new.columns:
        raise SystemExit(f"Missing {args.score_col} in inputs.")

    merged = old[["id", args.score_col]].merge(
        new[["id", args.score_col]],
        on="id",
        how="inner",
        suffixes=("_old", "_new"),
    )
    if merged.empty:
        raise SystemExit("No overlapping IDs between old and new predictions.")

    old_vals = merged[f"{args.score_col}_old"].to_numpy()
    new_vals = merged[f"{args.score_col}_new"].to_numpy()

    # Top-k overlap
    top_old = set(
        old.sort_values(args.score_col, ascending=False).head(args.topk)["id"]
    )
    top_new = set(
        new.sort_values(args.score_col, ascending=False).head(args.topk)["id"]
    )
    overlap = len(top_old & top_new)

    summary = {
        "n_overlap": len(merged),
        "old_mean": float(np.mean(old_vals)),
        "old_median": float(np.median(old_vals)),
        "old_std": float(np.std(old_vals, ddof=1)),
        "new_mean": float(np.mean(new_vals)),
        "new_median": float(np.median(new_vals)),
        "new_std": float(np.std(new_vals, ddof=1)),
        "topk": args.topk,
        "topk_overlap": overlap,
        "topk_overlap_rate": overlap / args.topk if args.topk else 0.0,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)
    merged.to_csv(args.out_dir / "predictions_merged.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    # Histogram overlay
    plt.figure(figsize=(8, 5))
    plt.hist(old_vals, bins=40, alpha=0.6, label="old (optimistic)")
    plt.hist(new_vals, bins=40, alpha=0.6, label="new (pessimistic, PETase FT)")
    plt.xlabel(args.score_col)
    plt.ylabel("count")
    plt.title("Prediction distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "hist_compare.png", dpi=180)
    plt.close()

    # ECDF
    x_old, y_old = ecdf(old_vals)
    x_new, y_new = ecdf(new_vals)
    plt.figure(figsize=(7, 5))
    plt.plot(x_old, y_old, label="old")
    plt.plot(x_new, y_new, label="new")
    plt.xlabel(args.score_col)
    plt.ylabel("ECDF")
    plt.title("ECDF comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "ecdf_compare.png", dpi=180)
    plt.close()

    # Scatter old vs new
    plt.figure(figsize=(6, 6))
    plt.scatter(old_vals, new_vals, s=8, alpha=0.5)
    lims = [
        min(old_vals.min(), new_vals.min()),
        max(old_vals.max(), new_vals.max()),
    ]
    plt.plot(lims, lims, "--", color="gray", linewidth=1)
    plt.xlabel("old (optimistic)")
    plt.ylabel("new (pessimistic)")
    plt.title("Old vs new predictions")
    plt.tight_layout()
    plt.savefig(args.out_dir / "scatter_old_vs_new.png", dpi=180)
    plt.close()

    # Rank plot (top 500)
    n = min(500, len(old))
    plt.figure(figsize=(8, 5))
    plt.plot(
        old.sort_values(args.score_col, ascending=False)[args.score_col].head(n).values,
        label="old",
    )
    plt.plot(
        new.sort_values(args.score_col, ascending=False)[args.score_col].head(n).values,
        label="new",
    )
    plt.xlabel("rank")
    plt.ylabel(args.score_col)
    plt.title("Top-ranked comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "rank_top500.png", dpi=180)
    plt.close()

    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
