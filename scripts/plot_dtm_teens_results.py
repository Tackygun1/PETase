#!/usr/bin/env python3
"""
Generate detailed publication-style figures for WT-centered Delta Tm predictions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot detailed Delta Tm figures for slideshow/publication use."
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/predictions_calibrated.csv"),
        help="CSV with WT-centered calibrated Delta Tm predictions.",
    )
    p.add_argument(
        "--pred-col",
        type=str,
        default="prediction_calibrated_wt_centered",
        help="Column name containing WT-centered Delta Tm values.",
    )
    p.add_argument(
        "--consensus",
        type=Path,
        default=Path("results/surrogate_af2/consensus/predictions_consensus.csv"),
        help="Consensus CSV with dtm_value and tm_value for correlation plot.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/figures_dtm_teens"),
        help="Output directory for figures and summary tables.",
    )
    p.add_argument(
        "--teen-threshold",
        type=float,
        default=13.0,
        help="Lower threshold for strict teen-range focus.",
    )
    p.add_argument(
        "--high-threshold",
        type=float,
        default=10.0,
        help="Lower threshold for high Delta Tm enrichment.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top candidates to show in bar chart.",
    )
    return p.parse_args()


def load_dtm_predictions(path: Path, col: str) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    vals: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or col not in reader.fieldnames:
            raise SystemExit(f"Missing required columns id/{col} in {path}")
        for row in reader:
            rid = (row.get("id") or "").strip()
            raw = (row.get(col) or "").strip()
            if not rid or not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            ids.append(rid)
            vals.append(value)
    if not ids:
        raise SystemExit(f"No valid Delta Tm rows in {path}")
    return ids, np.array(vals, dtype=float)


def load_consensus(path: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        needed = {"id", "dtm_value", "tm_value"}
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Missing required columns {needed} in {path}")
        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            try:
                dtm = float((row.get("dtm_value") or "").strip())
                tm = float((row.get("tm_value") or "").strip())
            except ValueError:
                continue
            out[rid] = (dtm, tm)
    return out


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    sorted_vals = a[order]
    ranks = np.empty(a.size, dtype=float)
    start = 0
    while start < a.size:
        end = start + 1
        while end < a.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson_r(rx, ry)


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_path: Path,
    dtm: np.ndarray,
    high_thr: float,
    teen_thr: float,
) -> None:
    q1, q3 = np.percentile(dtm, [25, 75])
    rows = [
        ("n", float(dtm.size)),
        ("mean", float(np.mean(dtm))),
        ("std", float(np.std(dtm, ddof=1))),
        ("median", float(np.median(dtm))),
        ("q1", float(q1)),
        ("q3", float(q3)),
        ("min", float(np.min(dtm))),
        ("max", float(np.max(dtm))),
        (f"n_ge_{high_thr:g}", float(np.sum(dtm >= high_thr))),
        (f"frac_ge_{high_thr:g}", float(np.mean(dtm >= high_thr))),
        (f"n_ge_{teen_thr:g}", float(np.sum(dtm >= teen_thr))),
        (f"frac_ge_{teen_thr:g}", float(np.mean(dtm >= teen_thr))),
        (f"n_between_{high_thr:g}_and_20", float(np.sum((dtm >= high_thr) & (dtm < 20.0)))),
        (f"n_between_{teen_thr:g}_and_20", float(np.sum((dtm >= teen_thr) & (dtm < 20.0)))),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def plot_distribution(
    dtm: np.ndarray,
    out_stem: Path,
    high_thr: float,
    teen_thr: float,
) -> None:
    mu = float(np.mean(dtm))
    sigma = float(np.std(dtm, ddof=1))
    med = float(np.median(dtm))
    high_n = int(np.sum(dtm >= high_thr))
    teen_n = int(np.sum((dtm >= teen_thr) & (dtm < 20.0)))
    bins = np.linspace(np.min(dtm) - 0.5, np.max(dtm) + 0.5, 55)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(dtm, bins=bins, color="#4C78A8", alpha=0.88, edgecolor="white", linewidth=0.7)

    ax.axvline(mu, color="#1F2D3D", linestyle="-", linewidth=2, label=fr"$\mu={mu:.2f}^\circ$C")
    ax.axvline(med, color="#2CA02C", linestyle="--", linewidth=2, label=fr"median={med:.2f}$^\circ$C")
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=2, label=fr"${high_thr:g}^\circ$C threshold")
    ax.axvline(teen_thr, color="#D62728", linestyle="-.", linewidth=2, label=fr"{teen_thr:g}$^\circ$C teen threshold")

    ax.set_title(r"Distribution of Predicted WT-Centered $\Delta T_m$", fontsize=14, pad=12)
    ax.set_xlabel(r"Predicted WT-centered $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel("Candidate count (n)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    textbox = (
        f"n = {dtm.size}\n"
        f"$\\mu \\pm \\sigma$ = {mu:.2f} ± {sigma:.2f} $^\\circ$C\n"
        f"$\\Delta T_m \\geq {high_thr:g}^\\circ$C: {high_n} ({100*high_n/dtm.size:.1f}%)\n"
        f"{teen_thr:g}$^\\circ$C ≤ $\\Delta T_m$ < 20$^\\circ$C: {teen_n}"
    )
    ax.text(
        0.985,
        0.97,
        textbox,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )
    save_figure(fig, out_stem)


def plot_ecdf(
    dtm: np.ndarray,
    out_stem: Path,
    high_thr: float,
    teen_thr: float,
) -> None:
    x = np.sort(dtm)
    y = np.arange(1, x.size + 1) / x.size

    frac_ge_high = float(np.mean(dtm >= high_thr))
    frac_ge_teen = float(np.mean(dtm >= teen_thr))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color="#4C78A8", linewidth=2.2, label="Empirical CDF")
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=1.8)
    ax.axvline(teen_thr, color="#D62728", linestyle="-.", linewidth=1.8)
    ax.axhline(1.0 - frac_ge_high, color="#F28E2B", linestyle=":", linewidth=1.5)
    ax.axhline(1.0 - frac_ge_teen, color="#D62728", linestyle=":", linewidth=1.5)

    ax.set_title(r"Empirical CDF of Predicted WT-Centered $\Delta T_m$", fontsize=14, pad=12)
    ax.set_xlabel(r"Predicted WT-centered $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel("Cumulative fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle="--", alpha=0.3)

    text = (
        f"P($\\Delta T_m \\geq {high_thr:g}^\\circ$C) = {frac_ge_high:.3f}\n"
        f"P($\\Delta T_m \\geq {teen_thr:g}^\\circ$C) = {frac_ge_teen:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, out_stem)


def plot_rank_curve(
    ids: Sequence[str],
    dtm: np.ndarray,
    out_stem: Path,
    high_thr: float,
    teen_thr: float,
) -> None:
    order = np.argsort(-dtm)
    sorted_vals = dtm[order]
    sorted_ids = [ids[i] for i in order]
    rank = np.arange(1, sorted_vals.size + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rank, sorted_vals, color="#1F77B4", linewidth=1.8, label="Ranked candidates")
    ax.fill_between(
        rank,
        0,
        sorted_vals,
        where=sorted_vals >= high_thr,
        color="#F28E2B",
        alpha=0.15,
        label=fr"$\Delta T_m \geq {high_thr:g}^\circ$C",
    )
    ax.fill_between(
        rank,
        0,
        sorted_vals,
        where=sorted_vals >= teen_thr,
        color="#D62728",
        alpha=0.12,
        label=fr"$\Delta T_m \geq {teen_thr:g}^\circ$C",
    )

    for i in range(min(8, sorted_vals.size)):
        ax.scatter(rank[i], sorted_vals[i], color="#D62728", s=24, zorder=4)
        ax.annotate(
            sorted_ids[i],
            (rank[i], sorted_vals[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#333333",
        )

    ax.set_title(r"Rank-Ordered Predicted WT-Centered $\Delta T_m$", fontsize=14, pad=12)
    ax.set_xlabel(r"Candidate rank (descending predicted $\Delta T_m$)")
    ax.set_ylabel(r"Predicted WT-centered $\Delta T_m$ ($^\circ$C)")
    ax.set_xlim(1, sorted_vals.size)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, out_stem)


def plot_scatter_dtm_tm(
    consensus: Dict[str, Tuple[float, float]],
    out_stem: Path,
) -> None:
    ids = sorted(consensus.keys())
    dtm = np.array([consensus[i][0] for i in ids], dtype=float)
    tm = np.array([consensus[i][1] for i in ids], dtype=float)

    slope, intercept = np.polyfit(dtm, tm, 1)
    x_line = np.linspace(np.min(dtm), np.max(dtm), 200)
    y_line = slope * x_line + intercept
    r = pearson_r(dtm, tm)
    rho = spearman_rho(dtm, tm)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(dtm, tm, s=16, alpha=0.55, color="#4C78A8", edgecolor="white", linewidth=0.25)
    ax.plot(x_line, y_line, color="#D62728", linewidth=2.0, label="Least-squares fit")

    ax.set_title(r"Predicted $\Delta T_m$ vs Predicted $T_m$", fontsize=14, pad=12)
    ax.set_xlabel(r"Predicted WT-centered $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel(r"Predicted $T_m$ ($^\circ$C)")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    stats_text = (
        f"n = {dtm.size}\n"
        f"$r$ = {r:.3f}\n"
        f"$\\rho$ = {rho:.3f}\n"
        f"$T_m$ = {slope:.3f}$\\cdot\\Delta T_m$ + {intercept:.3f}"
    )
    ax.text(
        0.985,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )
    save_figure(fig, out_stem)


def plot_top_bar(
    ids: Sequence[str],
    dtm: np.ndarray,
    out_stem: Path,
    top_n: int,
    high_thr: float,
    teen_thr: float,
) -> None:
    order = np.argsort(-dtm)[:top_n]
    top_ids = [ids[i] for i in order][::-1]
    top_vals = dtm[order][::-1]
    ypos = np.arange(len(top_ids))

    fig_h = max(8, 0.28 * len(top_ids))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    colors = ["#D62728" if v >= teen_thr else "#F28E2B" if v >= high_thr else "#4C78A8" for v in top_vals]
    ax.barh(ypos, top_vals, color=colors, alpha=0.92, edgecolor="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels(top_ids, fontsize=8)
    ax.set_xlabel(r"Predicted WT-centered $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel("Candidate ID")
    ax.set_title(fr"Top {top_n} Candidates by Predicted WT-Centered $\Delta T_m$", fontsize=14, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=1.5)
    ax.axvline(teen_thr, color="#D62728", linestyle="-.", linewidth=1.5)

    x_pad = 0.01 * max(np.max(top_vals), 1.0)
    for y, v in zip(ypos, top_vals):
        ax.text(v + x_pad, y, f"{v:.2f}", va="center", fontsize=8)
    save_figure(fig, out_stem)


def write_teen_candidates(
    out_path: Path,
    ids: Sequence[str],
    dtm: np.ndarray,
    high_thr: float,
    teen_thr: float,
) -> None:
    order = np.argsort(-dtm)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "delta_tm_wt_centered_c", "tier"])
        for i in order:
            v = float(dtm[i])
            if v >= teen_thr:
                tier = f"ge_{teen_thr:g}"
            elif v >= high_thr:
                tier = f"ge_{high_thr:g}"
            else:
                continue
            writer.writerow([ids[i], f"{v:.6f}", tier])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    ids, dtm = load_dtm_predictions(args.predictions, args.pred_col)
    consensus = load_consensus(args.consensus)

    write_summary(
        args.out_dir / "summary_stats.csv",
        dtm=dtm,
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )
    write_teen_candidates(
        args.out_dir / "teen_candidates.csv",
        ids=ids,
        dtm=dtm,
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )

    plot_distribution(
        dtm=dtm,
        out_stem=args.out_dir / "fig1_dtm_distribution",
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )
    plot_ecdf(
        dtm=dtm,
        out_stem=args.out_dir / "fig2_dtm_ecdf",
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )
    plot_rank_curve(
        ids=ids,
        dtm=dtm,
        out_stem=args.out_dir / "fig3_dtm_rank_curve",
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )
    plot_scatter_dtm_tm(
        consensus=consensus,
        out_stem=args.out_dir / "fig4_dtm_vs_tm_scatter",
    )
    plot_top_bar(
        ids=ids,
        dtm=dtm,
        out_stem=args.out_dir / "fig5_top_dtm_candidates",
        top_n=args.top_n,
        high_thr=args.high_threshold,
        teen_thr=args.teen_threshold,
    )

    print(f"Wrote figures and summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
