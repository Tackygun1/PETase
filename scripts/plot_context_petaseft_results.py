#!/usr/bin/env python3
"""
Generate publication-style figures for PETase-FT context surrogate predictions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot detailed figures for PETase-FT context surrogate predictions."
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv"),
        help="CSV with PETase-FT context predictions.",
    )
    p.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="ID column name.",
    )
    p.add_argument(
        "--residual-col",
        type=str,
        default="pred_residual",
        help="Residual prediction column name.",
    )
    p.add_argument(
        "--absolute-col",
        type=str,
        default="pred_abs",
        help="Absolute prediction column name.",
    )
    p.add_argument(
        "--baseline-col",
        type=str,
        default="baseline",
        help="Baseline column name (optional for plotting/annotation).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/figures_context_petaseft"),
        help="Output directory for figures and summary tables.",
    )
    p.add_argument(
        "--residual-high-threshold",
        type=float,
        default=2.0,
        help="High threshold for residual values (deg C).",
    )
    p.add_argument(
        "--residual-elite-threshold",
        type=float,
        default=2.5,
        help="Elite threshold for residual values (deg C).",
    )
    p.add_argument(
        "--absolute-high-threshold",
        type=float,
        default=0.0,
        help="High threshold for absolute values (deg C).",
    )
    p.add_argument(
        "--absolute-elite-threshold",
        type=float,
        default=1.0,
        help="Elite threshold for absolute values (deg C).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top residual candidates shown in bar chart and table.",
    )
    return p.parse_args()


def load_predictions(
    path: Path,
    id_col: str,
    residual_col: str,
    absolute_col: str,
    baseline_col: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    ids: List[str] = []
    residuals: List[float] = []
    absolutes: List[float] = []
    baselines: List[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {path}")
        required = {id_col, residual_col, absolute_col}
        if not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Missing required columns {required} in {path}")

        has_baseline = baseline_col in set(reader.fieldnames)
        for row in reader:
            rid = (row.get(id_col) or "").strip()
            raw_res = (row.get(residual_col) or "").strip()
            raw_abs = (row.get(absolute_col) or "").strip()
            raw_base = (row.get(baseline_col) or "").strip() if has_baseline else ""

            if not rid or not raw_res or not raw_abs:
                continue
            try:
                res_val = float(raw_res)
                abs_val = float(raw_abs)
            except ValueError:
                continue

            if raw_base:
                try:
                    base_val = float(raw_base)
                except ValueError:
                    base_val = float("nan")
            else:
                base_val = float("nan")

            ids.append(rid)
            residuals.append(res_val)
            absolutes.append(abs_val)
            baselines.append(base_val)

    if not ids:
        raise SystemExit(f"No valid rows found in {path}")

    return (
        ids,
        np.array(residuals, dtype=float),
        np.array(absolutes, dtype=float),
        np.array(baselines, dtype=float),
    )


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
    values: np.ndarray,
    high_thr: float,
    elite_thr: float,
) -> None:
    q1, q3 = np.percentile(values, [25, 75])
    rows = [
        ("n", float(values.size)),
        ("mean", float(np.mean(values))),
        ("std", float(np.std(values, ddof=1))),
        ("median", float(np.median(values))),
        ("q1", float(q1)),
        ("q3", float(q3)),
        ("min", float(np.min(values))),
        ("max", float(np.max(values))),
        (f"n_ge_{high_thr:g}", float(np.sum(values >= high_thr))),
        (f"frac_ge_{high_thr:g}", float(np.mean(values >= high_thr))),
        (f"n_ge_{elite_thr:g}", float(np.sum(values >= elite_thr))),
        (f"frac_ge_{elite_thr:g}", float(np.mean(values >= elite_thr))),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def plot_distribution(
    values: np.ndarray,
    out_stem: Path,
    title: str,
    xlabel: str,
    high_thr: float,
    elite_thr: float,
    color: str,
) -> None:
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    med = float(np.median(values))
    n_high = int(np.sum(values >= high_thr))
    n_elite = int(np.sum(values >= elite_thr))
    bins = np.linspace(np.min(values) - 0.1, np.max(values) + 0.1, 55)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins, color=color, alpha=0.88, edgecolor="white", linewidth=0.7)
    ax.axvline(mu, color="#1F2D3D", linestyle="-", linewidth=2, label=fr"$\mu={mu:.3f}^\circ$C")
    ax.axvline(med, color="#2CA02C", linestyle="--", linewidth=2, label=fr"median={med:.3f}$^\circ$C")
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=2, label=fr"{high_thr:g}$^\circ$C threshold")
    ax.axvline(
        elite_thr,
        color="#D62728",
        linestyle="-.",
        linewidth=2,
        label=fr"{elite_thr:g}$^\circ$C elite threshold",
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Candidate count (n)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    textbox = (
        f"n = {values.size}\n"
        f"$\\mu \\pm \\sigma$ = {mu:.3f} ± {sigma:.3f} $^\\circ$C\n"
        f"count >= {high_thr:g}$^\\circ$C: {n_high} ({100*n_high/values.size:.2f}%)\n"
        f"count >= {elite_thr:g}$^\\circ$C: {n_elite} ({100*n_elite/values.size:.2f}%)"
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
    values: np.ndarray,
    out_stem: Path,
    title: str,
    xlabel: str,
    high_thr: float,
    elite_thr: float,
) -> None:
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    frac_ge_high = float(np.mean(values >= high_thr))
    frac_ge_elite = float(np.mean(values >= elite_thr))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color="#4C78A8", linewidth=2.2, label="Empirical CDF")
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=1.8)
    ax.axvline(elite_thr, color="#D62728", linestyle="-.", linewidth=1.8)
    ax.axhline(1.0 - frac_ge_high, color="#F28E2B", linestyle=":", linewidth=1.5)
    ax.axhline(1.0 - frac_ge_elite, color="#D62728", linestyle=":", linewidth=1.5)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    text = (
        f"P(value >= {high_thr:g}$^\\circ$C) = {frac_ge_high:.4f}\n"
        f"P(value >= {elite_thr:g}$^\\circ$C) = {frac_ge_elite:.4f}"
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
    save_figure(fig, out_stem)


def plot_rank_curve(
    ids: Sequence[str],
    values: np.ndarray,
    out_stem: Path,
    title: str,
    ylabel: str,
    high_thr: float,
    elite_thr: float,
    annotate_top: int = 10,
) -> None:
    order = np.argsort(-values)
    sorted_vals = values[order]
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
        label=fr"value >= {high_thr:g}$^\circ$C",
    )
    ax.fill_between(
        rank,
        0,
        sorted_vals,
        where=sorted_vals >= elite_thr,
        color="#D62728",
        alpha=0.12,
        label=fr"value >= {elite_thr:g}$^\circ$C",
    )

    for i in range(min(annotate_top, sorted_vals.size)):
        ax.scatter(rank[i], sorted_vals[i], color="#D62728", s=24, zorder=4)
        ax.annotate(
            sorted_ids[i],
            (rank[i], sorted_vals[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#333333",
        )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Candidate rank (descending)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, sorted_vals.size)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, out_stem)


def plot_residual_vs_absolute(
    residual: np.ndarray,
    absolute: np.ndarray,
    baselines: np.ndarray,
    out_stem: Path,
) -> None:
    slope, intercept = np.polyfit(residual, absolute, 1)
    x_line = np.linspace(np.min(residual), np.max(residual), 200)
    y_line = slope * x_line + intercept

    valid_baselines = baselines[np.isfinite(baselines)]
    if valid_baselines.size:
        base_min = float(np.min(valid_baselines))
        base_max = float(np.max(valid_baselines))
        base_med = float(np.median(valid_baselines))
    else:
        base_min = float("nan")
        base_max = float("nan")
        base_med = float("nan")

    r = pearson_r(residual, absolute)
    rho = spearman_rho(residual, absolute)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(residual, absolute, s=16, alpha=0.55, color="#4C78A8", edgecolor="white", linewidth=0.25)
    ax.plot(x_line, y_line, color="#D62728", linewidth=2.0, label="Least-squares fit")

    if np.isfinite(base_med):
        ax.plot(x_line, x_line + base_med, color="#2CA02C", linestyle="--", linewidth=1.8, label=fr"$y=x+{base_med:.2f}$")

    ax.set_title(r"PETase-FT Context: $pred\_residual$ vs $pred\_abs$", fontsize=14, pad=12)
    ax.set_xlabel(r"Predicted residual $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel(r"Predicted absolute-adjusted value ($^\circ$C)")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    stats_text = (
        f"n = {residual.size}\n"
        f"$r$ = {r:.4f}\n"
        f"$\\rho$ = {rho:.4f}\n"
        f"$y$ = {slope:.3f}$\\cdot x$ + {intercept:.3f}\n"
        f"baseline min/max = {base_min:.2f}, {base_max:.2f}"
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
    residual: np.ndarray,
    absolute: np.ndarray,
    out_stem: Path,
    top_n: int,
    high_thr: float,
    elite_thr: float,
) -> None:
    order = np.argsort(-residual)[:top_n]
    top_ids = [ids[i] for i in order][::-1]
    top_residual = residual[order][::-1]
    top_absolute = absolute[order][::-1]
    ypos = np.arange(len(top_ids))

    fig_h = max(8, 0.28 * len(top_ids))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    colors = ["#D62728" if v >= elite_thr else "#F28E2B" if v >= high_thr else "#4C78A8" for v in top_residual]
    ax.barh(ypos, top_residual, color=colors, alpha=0.92, edgecolor="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels(top_ids, fontsize=8)
    ax.set_xlabel(r"Predicted residual $\Delta T_m$ ($^\circ$C)")
    ax.set_ylabel("Candidate ID")
    ax.set_title(fr"Top {top_n} PETase-FT Context Candidates by Residual $\Delta T_m$", fontsize=14, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.axvline(high_thr, color="#F28E2B", linestyle="--", linewidth=1.5)
    ax.axvline(elite_thr, color="#D62728", linestyle="-.", linewidth=1.5)

    x_pad = 0.01 * max(np.max(top_residual), 1.0)
    for y, v_res, v_abs in zip(ypos, top_residual, top_absolute):
        ax.text(v_res + x_pad, y, f"res={v_res:.2f}, abs={v_abs:.2f}", va="center", fontsize=8)
    save_figure(fig, out_stem)


def write_top_candidates(
    out_path: Path,
    ids: Sequence[str],
    residual: np.ndarray,
    absolute: np.ndarray,
    baselines: np.ndarray,
    top_n: int,
) -> None:
    order = np.argsort(-residual)[:top_n]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "id", "pred_residual_c", "pred_abs_c", "baseline_c"])
        for rank, i in enumerate(order, start=1):
            base = baselines[i]
            base_out = "" if not np.isfinite(base) else f"{base:.6f}"
            writer.writerow([rank, ids[i], f"{residual[i]:.6f}", f"{absolute[i]:.6f}", base_out])


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

    ids, residual, absolute, baselines = load_predictions(
        path=args.predictions,
        id_col=args.id_col,
        residual_col=args.residual_col,
        absolute_col=args.absolute_col,
        baseline_col=args.baseline_col,
    )

    write_summary(
        args.out_dir / "summary_residual.csv",
        values=residual,
        high_thr=args.residual_high_threshold,
        elite_thr=args.residual_elite_threshold,
    )
    write_summary(
        args.out_dir / "summary_absolute.csv",
        values=absolute,
        high_thr=args.absolute_high_threshold,
        elite_thr=args.absolute_elite_threshold,
    )
    write_top_candidates(
        args.out_dir / "top_candidates.csv",
        ids=ids,
        residual=residual,
        absolute=absolute,
        baselines=baselines,
        top_n=args.top_n,
    )

    plot_distribution(
        values=residual,
        out_stem=args.out_dir / "fig1_residual_distribution",
        title=r"Distribution of PETase-FT Context Residual Predictions",
        xlabel=r"Predicted residual $\Delta T_m$ ($^\circ$C)",
        high_thr=args.residual_high_threshold,
        elite_thr=args.residual_elite_threshold,
        color="#4C78A8",
    )
    plot_ecdf(
        values=residual,
        out_stem=args.out_dir / "fig2_residual_ecdf",
        title=r"Empirical CDF of PETase-FT Context Residual Predictions",
        xlabel=r"Predicted residual $\Delta T_m$ ($^\circ$C)",
        high_thr=args.residual_high_threshold,
        elite_thr=args.residual_elite_threshold,
    )
    plot_rank_curve(
        ids=ids,
        values=residual,
        out_stem=args.out_dir / "fig3_residual_rank_curve",
        title=r"Rank-Ordered PETase-FT Context Residual Predictions",
        ylabel=r"Predicted residual $\Delta T_m$ ($^\circ$C)",
        high_thr=args.residual_high_threshold,
        elite_thr=args.residual_elite_threshold,
    )
    plot_residual_vs_absolute(
        residual=residual,
        absolute=absolute,
        baselines=baselines,
        out_stem=args.out_dir / "fig4_residual_vs_absolute",
    )
    plot_top_bar(
        ids=ids,
        residual=residual,
        absolute=absolute,
        out_stem=args.out_dir / "fig5_top_residual_candidates",
        top_n=args.top_n,
        high_thr=args.residual_high_threshold,
        elite_thr=args.residual_elite_threshold,
    )
    plot_distribution(
        values=absolute,
        out_stem=args.out_dir / "fig6_absolute_distribution",
        title=r"Distribution of PETase-FT Context Absolute-Adjusted Predictions",
        xlabel=r"Predicted absolute-adjusted value ($^\circ$C)",
        high_thr=args.absolute_high_threshold,
        elite_thr=args.absolute_elite_threshold,
        color="#72B7B2",
    )

    print(f"Wrote context PETase-FT figures and summaries to {args.out_dir}")


if __name__ == "__main__":
    main()

