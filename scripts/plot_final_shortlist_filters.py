#!/usr/bin/env python3
"""
Generate detailed figures for the multimodel final shortlist.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot final-shortlist multimodel filter statistics."
    )
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_with_af2_metrics.csv"),
        help="Merged shortlist table with AF2 metrics.",
    )
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv"),
        help="Context candidate predictions table (for starting pool count).",
    )
    p.add_argument(
        "--topn-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/topn.csv"),
        help="Top-N candidate table.",
    )
    p.add_argument(
        "--filtered-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/filtered_seq.csv"),
        help="Sequence-QC filtered table.",
    )
    p.add_argument(
        "--rosetta-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/rosetta/rosetta_ddg.csv"),
        help="Rosetta ddG output table.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/figures_final_shortlist"),
        help="Output directory for figures and summaries.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top candidates for top-candidate panel.",
    )
    p.add_argument(
        "--wt-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/wt_multimodel_evaluation.csv"),
        help="Optional WT multimodel evaluation CSV for reference overlays.",
    )
    return p.parse_args()


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


def compute_filter_counts(
    merged: pd.DataFrame,
    predictions_csv: Path,
    topn_csv: Path,
    filtered_csv: Path,
    rosetta_csv: Path,
) -> List[Tuple[str, int]]:
    n_all = count_rows(predictions_csv)
    n_topn = count_rows(topn_csv)
    n_seq_qc = count_rows(filtered_csv)
    n_rosetta_scored = count_rows(rosetta_csv)
    n_rosetta_pass = int(pd.Series(merged.get("rosetta_pass_ddg_le_0", False)).fillna(False).sum())
    n_af2_complete = int(pd.Series(merged.get("af2_complete", False)).fillna(False).sum())
    n_af2_pass = int(pd.Series(merged.get("af2_pass_all", False)).fillna(False).sum())
    n_final = int(pd.Series(merged.get("combined_structural_pass", False)).fillna(False).sum())

    return [
        ("Context candidate pool", n_all),
        ("Top-N selected", n_topn),
        ("Sequence QC pass", n_seq_qc),
        ("Rosetta scored", n_rosetta_scored),
        ("Rosetta ddG <= 0 REU", n_rosetta_pass),
        ("AF2 completed", n_af2_complete),
        ("AF2 quality+geometry pass", n_af2_pass),
        ("Final combined pass", n_final),
    ]


def write_filter_summary(path: Path, counts: Sequence[Tuple[str, int]]) -> None:
    rows = []
    base = counts[0][1] if counts else 0
    prev = None
    for stage, n in counts:
        pct_vs_all = (100.0 * n / base) if base else float("nan")
        pct_vs_prev = (100.0 * n / prev) if prev else float("nan")
        rows.append((stage, n, pct_vs_all, pct_vs_prev))
        prev = n
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "count", "pct_vs_all", "pct_vs_prev"])
        w.writerows(rows)


def plot_filter_funnel(counts: Sequence[Tuple[str, int]], out_stem: Path) -> None:
    stages = [x[0] for x in counts]
    vals = np.array([x[1] for x in counts], dtype=float)
    base = vals[0] if vals.size else np.nan

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ypos = np.arange(len(stages))
    colors = ["#4C78A8", "#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2", "#9D755D"]
    bars = ax.barh(ypos, vals, color=colors[: len(stages)], alpha=0.92, edgecolor="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate count (n)")
    ax.set_title("Multimodel Filter Funnel: Context -> Sequence QC -> Rosetta -> AlphaFold", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    prev = None
    for i, (b, v) in enumerate(zip(bars, vals)):
        pct_all = 100.0 * v / base if base else np.nan
        pct_prev = 100.0 * v / prev if (prev and prev > 0) else np.nan
        txt = f"{int(v)}"
        txt += f"\n({pct_all:.2f}% of pool)"
        if np.isfinite(pct_prev):
            txt += f"\n({pct_prev:.2f}% vs previous)"
        ax.text(
            b.get_width() + 0.008 * max(vals),
            b.get_y() + b.get_height() / 2.0,
            txt,
            va="center",
            fontsize=9,
        )
        prev = v

    save_figure(fig, out_stem)


def plot_residual_vs_rosetta(merged: pd.DataFrame, out_stem: Path) -> None:
    x = pd.to_numeric(merged["pred_residual"], errors="coerce").to_numpy()
    y = pd.to_numeric(merged["ddg_reu"], errors="coerce").to_numpy()
    ids = merged["id"].astype(str).to_list()

    slope, intercept = np.polyfit(x, y, 1)
    r = pearson_r(x, y)
    rho = spearman_rho(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 240)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.scatter(x, y, s=34, alpha=0.75, color="#4C78A8", edgecolor="white", linewidth=0.35)
    ax.plot(x_line, y_line, color="#D62728", linewidth=2.0, label="Least-squares fit")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, label=r"Rosetta threshold ($\Delta\Delta G = 0$ REU)")
    ax.set_xlabel(r"Context-model residual $\Delta T_m$ prediction ($^\circ$C)")
    ax.set_ylabel(r"Rosetta $\Delta\Delta G$ (REU)")
    ax.set_title(r"Context Residual vs Rosetta $\Delta\Delta G$ for Final Shortlist", fontsize=14, pad=12)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")

    stats_text = (
        f"n = {len(x)}\n"
        f"Pearson r = {r:.3f}\n"
        f"Spearman rho = {rho:.3f}\n"
        f"ddG = {slope:.2f}*residual + {intercept:.2f}"
    )
    ax.text(
        0.98,
        0.03,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )

    top_idx = np.argsort(-x)[:6]
    for i in top_idx:
        ax.annotate(
            ids[i],
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="#333333",
        )

    save_figure(fig, out_stem)


def plot_plddt_distribution(merged: pd.DataFrame, out_stem: Path) -> None:
    mean_plddt = pd.to_numeric(merged["af2_ranked0_plddt_mean"], errors="coerce").to_numpy()
    key_plddt = pd.to_numeric(merged["af2_ranked0_plddt_key_mean"], errors="coerce").to_numpy()

    bins = np.linspace(min(np.min(mean_plddt), np.min(key_plddt)) - 0.05, max(np.max(mean_plddt), np.max(key_plddt)) + 0.05, 22)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.hist(mean_plddt, bins=bins, alpha=0.70, color="#4C78A8", edgecolor="white", label=r"Ranked-0 mean pLDDT")
    ax.hist(key_plddt, bins=bins, alpha=0.62, color="#F58518", edgecolor="white", label=r"Ranked-0 key-site mean pLDDT")
    ax.axvline(90.0, color="#444444", linestyle="--", linewidth=1.4, label="90 threshold")
    ax.set_title("AlphaFold Confidence Distribution in Final Shortlist", fontsize=14, pad=12)
    ax.set_xlabel("pLDDT score (unitless)")
    ax.set_ylabel("Candidate count (n)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    txt = (
        f"n = {len(mean_plddt)}\n"
        f"mean pLDDT = {np.mean(mean_plddt):.2f} ± {np.std(mean_plddt, ddof=1):.2f}\n"
        f"key-site pLDDT = {np.mean(key_plddt):.2f} ± {np.std(key_plddt, ddof=1):.2f}"
    )
    ax.text(
        0.98,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )
    save_figure(fig, out_stem)


def plot_geometry_boxplots(merged: pd.DataFrame, out_stem: Path) -> None:
    triad_cols = [
        ("af2_triad_ser160_his237", "Ser160-His237"),
        ("af2_triad_asp206_his237", "Asp206-His237"),
        ("af2_triad_ser160_asp206", "Ser160-Asp206"),
    ]
    disulf_cols = [
        ("af2_disulfide_203_239_dist", "Cys203-Cys239"),
        ("af2_disulfide_273_289_dist", "Cys273-Cys289"),
    ]

    triad_data = [pd.to_numeric(merged[c], errors="coerce").dropna().to_numpy() for c, _ in triad_cols]
    dis_data = [pd.to_numeric(merged[c], errors="coerce").dropna().to_numpy() for c, _ in disulf_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.6))

    bp1 = ax1.boxplot(triad_data, patch_artist=True, tick_labels=[n for _, n in triad_cols], widths=0.58)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#72B7B2")
        patch.set_alpha(0.8)
    ax1.axhline(5.0, color="#D62728", linestyle="--", linewidth=1.5, label="5.0 A threshold")
    ax1.set_ylabel("Distance (A)")
    ax1.set_title("Catalytic-Triad Geometry")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, loc="upper right")

    bp2 = ax2.boxplot(dis_data, patch_artist=True, tick_labels=[n for _, n in disulf_cols], widths=0.58)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#54A24B")
        patch.set_alpha(0.8)
    ax2.axhline(2.8, color="#D62728", linestyle="--", linewidth=1.5, label="2.8 A threshold")
    ax2.set_ylabel("Distance (A)")
    ax2.set_title("Disulfide Geometry")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, loc="upper right")

    fig.suptitle("AlphaFold Structural Geometry Metrics (Ranked-0 Models)", fontsize=14, y=1.02)
    save_figure(fig, out_stem)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def zscore_value(value: float, ref: np.ndarray) -> float:
    if ref.size < 2:
        return float("nan")
    mu = float(np.mean(ref))
    sd = float(np.std(ref, ddof=1))
    if sd == 0.0:
        return 0.0
    return (float(value) - mu) / sd


def plot_ranked_profiles(
    merged: pd.DataFrame,
    out_stem: Path,
    wt_row: pd.Series | None = None,
) -> None:
    df = merged.sort_values("pred_residual", ascending=False).reset_index(drop=True)
    rank = np.arange(1, len(df) + 1)

    res_arr = pd.to_numeric(df["pred_residual"], errors="coerce").to_numpy()
    ros_arr = (-pd.to_numeric(df["ddg_reu"], errors="coerce")).to_numpy()
    pld_arr = pd.to_numeric(df["af2_ranked0_plddt_mean"], errors="coerce").to_numpy()

    z_res = zscore(res_arr)
    z_ros = zscore(ros_arr)
    z_pld = zscore(pld_arr)

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    ax.plot(rank, z_res, color="#4C78A8", linewidth=2.0, label=r"z(context residual $\Delta T_m$)")
    ax.plot(rank, z_ros, color="#F58518", linewidth=2.0, label=r"z(-Rosetta $\Delta\Delta G$)")
    ax.plot(rank, z_pld, color="#54A24B", linewidth=2.0, label="z(AFold pLDDT mean)")

    if wt_row is not None:
        wt_res = pd.to_numeric(pd.Series([wt_row.get("pred_residual", np.nan)]), errors="coerce").iloc[0]
        wt_ddg = pd.to_numeric(pd.Series([wt_row.get("ddg_reu_reference", np.nan)]), errors="coerce").iloc[0]
        wt_pld = pd.to_numeric(pd.Series([wt_row.get("af2_ranked0_plddt_mean", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(wt_res):
            ax.axhline(
                zscore_value(float(wt_res), res_arr),
                color="#4C78A8",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label=r"WT z(context residual $\Delta T_m$)",
            )
        if np.isfinite(wt_ddg):
            ax.axhline(
                zscore_value(float(-wt_ddg), ros_arr),
                color="#F58518",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label=r"WT z(-Rosetta $\Delta\Delta G$)",
            )
        if np.isfinite(wt_pld):
            ax.axhline(
                zscore_value(float(wt_pld), pld_arr),
                color="#54A24B",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label="WT z(AFold pLDDT mean)",
            )

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Candidate rank by context residual (descending)")
    ax.set_ylabel("Standard score (z)")
    ax.set_title("Ranked Multi-Metric Profile Across Final Shortlist", fontsize=14, pad=12)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, out_stem)


def plot_correlation_heatmap(merged: pd.DataFrame, out_stem: Path) -> None:
    cols = [
        "pred_residual",
        "pred_abs",
        "ddg_reu",
        "af2_ranked0_plddt_mean",
        "af2_ranked0_plddt_key_mean",
        "af2_triad_ser160_his237",
        "af2_triad_asp206_his237",
        "af2_disulfide_203_239_dist",
        "af2_disulfide_273_289_dist",
    ]
    label_map = {
        "pred_residual": "Residual dTm",
        "pred_abs": "Absolute dTm",
        "ddg_reu": "Rosetta ddG",
        "af2_ranked0_plddt_mean": "pLDDT mean",
        "af2_ranked0_plddt_key_mean": "pLDDT key mean",
        "af2_triad_ser160_his237": "Triad S160-H237",
        "af2_triad_asp206_his237": "Triad D206-H237",
        "af2_disulfide_203_239_dist": "Disulfide 203-239",
        "af2_disulfide_273_289_dist": "Disulfide 273-289",
    }

    num = merged[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels([label_map[c] for c in cols], rotation=35, ha="right")
    ax.set_yticklabels([label_map[c] for c in cols])
    ax.set_title("Spearman Correlation Matrix: Final Shortlist Metrics", fontsize=14, pad=12)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    save_figure(fig, out_stem)


def plot_top_candidates_dualbar(
    merged: pd.DataFrame,
    out_stem: Path,
    top_n: int,
    wt_row: pd.Series | None = None,
) -> None:
    df = merged.sort_values("pred_residual", ascending=False).head(top_n).copy()
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    height = 0.38

    fig_h = max(8.0, 0.33 * top_n)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    ax.barh(y + height / 2.0, df["pred_residual"], height=height, color="#4C78A8", label=r"Residual $\Delta T_m$ ($^\circ$C)")
    ax.barh(y - height / 2.0, -df["ddg_reu"], height=height, color="#F58518", label=r"$-\Delta\Delta G$ (REU)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["id"], fontsize=8)
    ax.set_xlabel("Metric magnitude (higher is better in both plotted directions)")
    ax.set_title(f"Top {top_n} Final Candidates: Context Residual vs Rosetta Magnitude", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if wt_row is not None:
        wt_res = pd.to_numeric(pd.Series([wt_row.get("pred_residual", np.nan)]), errors="coerce").iloc[0]
        wt_ddg = pd.to_numeric(pd.Series([wt_row.get("ddg_reu_reference", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(wt_res):
            ax.axvline(
                float(wt_res),
                color="#4C78A8",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=rf"WT residual $\Delta T_m$ = {float(wt_res):.2f} $^\circ$C",
            )
        if np.isfinite(wt_ddg):
            ax.axvline(
                float(-wt_ddg),
                color="#F58518",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=rf"WT $-\Delta\Delta G$ = {float(-wt_ddg):.2f} REU",
            )

    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, out_stem)


def plot_af2_timing(merged: pd.DataFrame, out_stem: Path) -> None:
    feat = pd.to_numeric(merged["af2_timing_features_s"], errors="coerce").to_numpy()
    pred_total = pd.to_numeric(merged["af2_timing_predict_total_s"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.scatter(feat, pred_total, s=30, color="#B279A2", alpha=0.75, edgecolor="white", linewidth=0.35)
    ax.set_xlabel("AF2 feature generation time (s)")
    ax.set_ylabel("AF2 total prediction+compile time over 5 models (s)")
    ax.set_title("AlphaFold Runtime Profile Per Final Candidate", fontsize=14, pad=12)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_yscale("log")
    txt = (
        f"n = {len(feat)}\n"
        f"feature time median = {np.median(feat):.1f} s\n"
        f"predict total median = {np.median(pred_total):.1f} s"
    )
    ax.text(
        0.98,
        0.03,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )
    save_figure(fig, out_stem)


def write_metric_summary(path: Path, merged: pd.DataFrame) -> None:
    metrics = [
        "pred_residual",
        "pred_abs",
        "ddg_reu",
        "af2_ranked0_plddt_mean",
        "af2_ranked0_plddt_key_mean",
        "af2_triad_ser160_his237",
        "af2_triad_asp206_his237",
        "af2_triad_ser160_asp206",
        "af2_disulfide_203_239_dist",
        "af2_disulfide_273_289_dist",
        "af2_timing_features_s",
        "af2_timing_predict_total_s",
    ]
    rows = []
    for m in metrics:
        s = pd.to_numeric(merged[m], errors="coerce")
        rows.append((m, float(s.mean()), float(s.std(ddof=1)), float(s.min()), float(s.max())))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "min", "max"])
        w.writerows(rows)


def write_top_table(path: Path, merged: pd.DataFrame, top_n: int) -> None:
    cols = [
        "id",
        "pred_residual",
        "pred_abs",
        "ddg_reu",
        "af2_ranked0_plddt_mean",
        "af2_ranked0_plddt_key_mean",
        "af2_triad_ser160_his237",
        "af2_triad_asp206_his237",
        "af2_disulfide_203_239_dist",
        "af2_disulfide_273_289_dist",
    ]
    merged.sort_values("pred_residual", ascending=False).head(top_n)[cols].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    merged = pd.read_csv(args.merged_csv)
    wt_row = None
    if args.wt_csv.exists():
        wt_df = pd.read_csv(args.wt_csv)
        if len(wt_df):
            wt_row = wt_df.iloc[0]

    counts = compute_filter_counts(
        merged=merged,
        predictions_csv=args.predictions_csv,
        topn_csv=args.topn_csv,
        filtered_csv=args.filtered_csv,
        rosetta_csv=args.rosetta_csv,
    )

    write_filter_summary(args.out_dir / "summary_filter_counts.csv", counts)
    write_metric_summary(args.out_dir / "summary_metric_stats.csv", merged)
    write_top_table(args.out_dir / "top_candidates_for_slides.csv", merged, args.top_n)

    plot_filter_funnel(counts, args.out_dir / "fig1_filter_funnel_counts")
    plot_residual_vs_rosetta(merged, args.out_dir / "fig2_residual_vs_rosetta")
    plot_plddt_distribution(merged, args.out_dir / "fig3_af2_plddt_distribution")
    plot_geometry_boxplots(merged, args.out_dir / "fig4_geometry_boxplots")
    plot_ranked_profiles(merged, args.out_dir / "fig5_ranked_zscore_profiles", wt_row=wt_row)
    plot_correlation_heatmap(merged, args.out_dir / "fig6_metric_correlation_heatmap")
    plot_top_candidates_dualbar(merged, args.out_dir / "fig7_top_candidates_dualbar", args.top_n, wt_row=wt_row)
    plot_af2_timing(merged, args.out_dir / "fig8_af2_timing_distribution")

    print(f"Wrote final-shortlist figure set to {args.out_dir}")


if __name__ == "__main__":
    main()
