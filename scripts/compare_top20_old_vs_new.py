#!/usr/bin/env python3
"""
Compare top-20 from old non-context surrogate vs new context PETase-FT surrogate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare old vs new top-20 candidates.")
    p.add_argument(
        "--old",
        type=Path,
        default=Path("results/surrogate_af2/predictions_with_af2.csv"),
        help="Old surrogate predictions with AF2 metrics.",
    )
    p.add_argument(
        "--old-score-col",
        default="prediction",
        help="Old surrogate score column (default: prediction).",
    )
    p.add_argument(
        "--new",
        type=Path,
        default=Path("results/surrogate_af2/context_top20_100k_petaseft.csv"),
        help="New PETase-FT top-20 CSV.",
    )
    p.add_argument(
        "--new-score-col",
        default="pred_residual",
        help="New surrogate score column (default: pred_residual).",
    )
    p.add_argument("--topk", type=int, default=20)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/compare_top20_old_vs_new"),
        help="Output directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    old = pd.read_csv(args.old)
    new = pd.read_csv(args.new)

    if args.old_score_col not in old.columns:
        raise SystemExit(f"Missing {args.old_score_col} in {args.old}")
    if args.new_score_col not in new.columns:
        raise SystemExit(f"Missing {args.new_score_col} in {args.new}")

    old_top = old.sort_values(args.old_score_col, ascending=False).head(args.topk).copy()
    new_top = new.sort_values(args.new_score_col, ascending=False).head(args.topk).copy()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old_top.to_csv(out_dir / "old_top20.csv", index=False)
    new_top.to_csv(out_dir / "new_top20.csv", index=False)

    # Overlap stats
    overlap = len(set(old_top["id"]) & set(new_top["id"]))
    summary = {
        "old_topk_mean": float(old_top[args.old_score_col].mean()),
        "old_topk_median": float(old_top[args.old_score_col].median()),
        "new_topk_mean": float(new_top[args.new_score_col].mean()),
        "new_topk_median": float(new_top[args.new_score_col].median()),
        "topk_overlap": overlap,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    # Score distributions (boxplots)
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [old_top[args.old_score_col].values, new_top[args.new_score_col].values],
        labels=["old top20", "new top20"],
    )
    plt.ylabel("score")
    plt.title("Top-20 score distributions")
    plt.tight_layout()
    plt.savefig(out_dir / "box_top20_scores.png", dpi=180)
    plt.close()

    # Ranked score curves
    plt.figure(figsize=(7, 5))
    plt.plot(
        old_top[args.old_score_col].sort_values(ascending=False).values,
        label="old top20",
    )
    plt.plot(
        new_top[args.new_score_col].sort_values(ascending=False).values,
        label="new top20",
    )
    plt.xlabel("rank")
    plt.ylabel("score")
    plt.title("Top-20 ranked scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rank_top20_scores.png", dpi=180)
    plt.close()

    # AF2 metrics for old top20 (if available)
    if {"plddt_mean", "plddt_min", "pae_mean"}.issubset(old_top.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(
            old_top[args.old_score_col],
            old_top["plddt_mean"],
            s=24,
            alpha=0.7,
        )
        plt.xlabel("old score")
        plt.ylabel("pLDDT mean")
        plt.title("Old top20: score vs pLDDT")
        plt.tight_layout()
        plt.savefig(out_dir / "old_top20_score_vs_plddt.png", dpi=180)
        plt.close()

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
