#!/usr/bin/env python3
"""
Create deep-dive analyses for the multimodel final shortlist:
1) Selection landscape with WT reference
2) Mutation-frequency map and recurring substitutions
3) AF2 model-to-model pLDDT spread
4) Abstract-ready summary tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTECTED_POSITIONS = (87, 160, 161, 185, 203, 206, 237, 239, 273, 289)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deep-dive shortlist figures + summary tables.")
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_with_af2_metrics.csv"),
        help="Merged final shortlist table.",
    )
    p.add_argument(
        "--shortlist-fasta",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist.fasta"),
        help="Final shortlist FASTA.",
    )
    p.add_argument(
        "--wt-fasta",
        type=Path,
        default=Path("data/processed/petase_wt.fasta"),
        help="WT FASTA.",
    )
    p.add_argument(
        "--wt-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/wt_multimodel_evaluation.csv"),
        help="WT multimodel evaluation CSV.",
    )
    p.add_argument(
        "--wt-percentiles-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/wt_vs_shortlist_percentiles.csv"),
        help="WT percentile-vs-shortlist CSV.",
    )
    p.add_argument(
        "--surrogate-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_surrogate_predictions.csv"),
        help="Surrogate predictions for shortlist IDs.",
    )
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv"),
        help="Context candidate predictions table (for pool count).",
    )
    p.add_argument(
        "--topn-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/topn.csv"),
        help="Top-N table.",
    )
    p.add_argument(
        "--filtered-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/filtered_seq.csv"),
        help="Sequence-filtered table.",
    )
    p.add_argument(
        "--rosetta-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/rosetta/rosetta_ddg.csv"),
        help="Rosetta-scored table.",
    )
    p.add_argument(
        "--label-top-n",
        type=int,
        default=12,
        help="Number of top residual candidates to label in the landscape plot.",
    )
    p.add_argument(
        "--uncertainty-top-n",
        type=int,
        default=20,
        help="Number of top residual candidates in AF2 spread plot.",
    )
    p.add_argument(
        "--top-substitutions",
        type=int,
        default=20,
        help="Number of recurring substitutions in the substitution panel.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/figures_final_shortlist"),
        help="Output directory.",
    )
    return p.parse_args()


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    if not path.exists():
        return seqs
    current_id = None
    parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(parts)
                current_id = line[1:].strip().split()[0]
                parts = []
            else:
                parts.append(line)
    if current_id is not None:
        seqs[current_id] = "".join(parts)
    return seqs


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


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


def prepare_merged_with_surrogate(merged: pd.DataFrame, surrogate_csv: Path) -> pd.DataFrame:
    out = merged.copy()
    if surrogate_csv.exists():
        surr = pd.read_csv(surrogate_csv)
        keep = [c for c in ["id", "surrogate_dtm_wt_centered_cal"] if c in surr.columns]
        if keep:
            out = out.merge(surr[keep], on="id", how="left")
    return out


def load_wt_row(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df.iloc[0]


def plot_selection_landscape(
    merged: pd.DataFrame,
    wt_row: pd.Series | None,
    out_stem: Path,
    label_top_n: int,
) -> None:
    x = pd.to_numeric(merged["pred_residual"], errors="coerce").to_numpy(dtype=float)
    y = (-pd.to_numeric(merged["ddg_reu"], errors="coerce")).to_numpy(dtype=float)
    c = pd.to_numeric(merged["af2_ranked0_plddt_mean"], errors="coerce").to_numpy(dtype=float)
    ids = merged["id"].astype(str).to_list()

    fig, ax = plt.subplots(figsize=(9.2, 7.1))
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=62,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
        label="Final shortlist candidates",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AF2 ranked-0 mean pLDDT (unitless)")

    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, label=r"$\Delta\Delta G = 0$ REU threshold")

    if wt_row is not None:
        wt_x = float(pd.to_numeric(pd.Series([wt_row.get("pred_residual", np.nan)]), errors="coerce").iloc[0])
        wt_ddg = float(pd.to_numeric(pd.Series([wt_row.get("ddg_reu_reference", np.nan)]), errors="coerce").iloc[0])
        wt_y = -wt_ddg
        wt_c = float(pd.to_numeric(pd.Series([wt_row.get("af2_ranked0_plddt_mean", np.nan)]), errors="coerce").iloc[0])
        if np.isfinite(wt_x) and np.isfinite(wt_y):
            ax.scatter(
                [wt_x],
                [wt_y],
                c=[wt_c if np.isfinite(wt_c) else np.nanmean(c)],
                cmap="viridis",
                marker="*",
                s=270,
                edgecolor="black",
                linewidth=1.0,
                label="WT reference",
                zorder=4,
            )
            ax.annotate(
                "WT",
                (wt_x, wt_y),
                textcoords="offset points",
                xytext=(7, 7),
                fontsize=9,
                fontweight="bold",
            )

    top_idx = np.argsort(-x)[: min(label_top_n, len(x))]
    for i in top_idx:
        sid = ids[i].replace("IsPETaseWT_pool_", "pool_")
        ax.annotate(
            sid,
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#222222",
        )

    r = pearson_r(x, y)
    rho = spearman_rho(x, y)
    ax.text(
        0.02,
        0.98,
        f"n = {len(x)}\nPearson r = {r:.3f}\nSpearman rho = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888888"),
    )

    ax.set_xlabel(r"Context-model residual $\Delta T_m$ prediction ($^\circ$C)")
    ax.set_ylabel(r"$-\Delta\Delta G$ (REU)")
    ax.set_title("Selection Landscape of Final Shortlist with WT Reference", fontsize=14, pad=12)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, out_stem)


def mutation_stats(
    wt_seq: str,
    shortlist_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not shortlist_map:
        return pd.DataFrame(), pd.DataFrame()
    L = len(wt_seq)
    pos_counts = np.zeros(L, dtype=int)
    sub_counts: Dict[str, int] = {}
    n_seq = 0
    for _, seq in shortlist_map.items():
        if len(seq) != L:
            continue
        n_seq += 1
        for i, (a, b) in enumerate(zip(wt_seq, seq), start=1):
            if a != b:
                pos_counts[i - 1] += 1
                key = f"{a}{i}{b}"
                sub_counts[key] = sub_counts.get(key, 0) + 1

    pos_df = pd.DataFrame(
        {
            "position": np.arange(1, L + 1),
            "wt_aa": list(wt_seq),
            "mutated_count": pos_counts,
            "mutation_frequency_pct": (100.0 * pos_counts / max(n_seq, 1)),
        }
    )
    sub_df = pd.DataFrame(
        sorted(
            (
                {"substitution": k, "count": v, "frequency_pct": 100.0 * v / max(n_seq, 1)}
                for k, v in sub_counts.items()
            ),
            key=lambda r: (-r["count"], r["substitution"]),
        )
    )
    return pos_df, sub_df


def plot_mutation_map(
    pos_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    out_stem: Path,
    top_substitutions: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12.2, 8.8),
        gridspec_kw={"height_ratios": [2.1, 1.6]},
    )

    ax1.bar(
        pos_df["position"].to_numpy(),
        pos_df["mutation_frequency_pct"].to_numpy(),
        color="#4C78A8",
        alpha=0.9,
        width=0.95,
    )
    for p in PROTECTED_POSITIONS:
        ax1.axvline(p, color="#D62728", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_xlim(0.5, float(pos_df["position"].max()) + 0.5)
    ax1.set_xlabel("Sequence position (1-indexed)")
    ax1.set_ylabel("Mutation frequency (%)")
    ax1.set_title("Final-Shortlist Mutation Frequency Along WT PETase Sequence")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.text(
        0.99,
        0.97,
        "Red dashed lines: protected positions",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#888888"),
    )

    top_sub = sub_df.head(top_substitutions).iloc[::-1].copy()
    if len(top_sub) > 0:
        ax2.barh(
            top_sub["substitution"].to_numpy(),
            top_sub["frequency_pct"].to_numpy(),
            color="#F58518",
            alpha=0.9,
        )
    ax2.set_xlabel("Frequency in final shortlist (%)")
    ax2.set_ylabel("Substitution")
    ax2.set_title(f"Top {top_substitutions} Recurring Substitutions")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Mutation Landscape of Final Shortlist Candidates", fontsize=14, y=1.01)
    save_figure(fig, out_stem)


def plot_af2_uncertainty(
    merged: pd.DataFrame,
    wt_row: pd.Series | None,
    out_stem: Path,
    top_n: int,
) -> None:
    df = merged.sort_values("pred_residual", ascending=False).head(top_n).copy()
    df = df.iloc[::-1].reset_index(drop=True)

    sid = df["id"].astype(str).to_numpy()
    mean = pd.to_numeric(df["af2_model_plddt_mean_mean"], errors="coerce").to_numpy(dtype=float)
    vmin = pd.to_numeric(df["af2_model_plddt_mean_min"], errors="coerce").to_numpy(dtype=float)
    vmax = pd.to_numeric(df["af2_model_plddt_mean_max"], errors="coerce").to_numpy(dtype=float)
    ranked0 = pd.to_numeric(df["af2_ranked0_plddt_mean"], errors="coerce").to_numpy(dtype=float)
    keymean = pd.to_numeric(df["af2_ranked0_plddt_key_mean"], errors="coerce").to_numpy(dtype=float)

    y = np.arange(len(df))
    xerr = np.vstack([mean - vmin, vmax - mean])

    fig_h = max(8.4, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    ax.errorbar(
        mean,
        y,
        xerr=xerr,
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=1.4,
        capsize=3.0,
        markersize=5.5,
        label="AF2 model-mean pLDDT (mean ± min/max over 5 models)",
    )
    ax.scatter(ranked0, y, color="#F58518", s=26, label="Ranked-0 mean pLDDT", zorder=3)
    ax.scatter(keymean, y, color="#54A24B", s=26, label="Ranked-0 key-site mean pLDDT", zorder=3)
    ax.axvline(90.0, color="#444444", linestyle="--", linewidth=1.2, label="pLDDT threshold = 90")

    if wt_row is not None:
        wt_mean = float(pd.to_numeric(pd.Series([wt_row.get("af2_model_plddt_mean_mean", np.nan)]), errors="coerce").iloc[0])
        wt_min = float(pd.to_numeric(pd.Series([wt_row.get("af2_model_plddt_mean_min", np.nan)]), errors="coerce").iloc[0])
        wt_max = float(pd.to_numeric(pd.Series([wt_row.get("af2_model_plddt_mean_max", np.nan)]), errors="coerce").iloc[0])
        if np.isfinite(wt_min) and np.isfinite(wt_max):
            ax.axvspan(wt_min, wt_max, color="#B279A2", alpha=0.15, label="WT AF2 model range")
        if np.isfinite(wt_mean):
            ax.axvline(wt_mean, color="#B279A2", linestyle="--", linewidth=1.4, label="WT AF2 model-mean")

    labels = [s.replace("IsPETaseWT_pool_", "pool_") for s in sid]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pLDDT score (unitless)")
    ax.set_title(f"AF2 Confidence Spread Across Top {len(df)} Candidates", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, out_stem)


def write_summary_tables(
    merged: pd.DataFrame,
    wt_row: pd.Series | None,
    wt_percentiles_csv: Path,
    predictions_csv: Path,
    topn_csv: Path,
    filtered_csv: Path,
    rosetta_csv: Path,
    out_dir: Path,
) -> None:
    n_all = count_rows(predictions_csv)
    n_topn = count_rows(topn_csv)
    n_seq_qc = count_rows(filtered_csv)
    n_rosetta_scored = count_rows(rosetta_csv)
    n_rosetta_pass = int(pd.Series(merged.get("rosetta_pass_ddg_le_0", False)).fillna(False).sum())
    n_af2_complete = int(pd.Series(merged.get("af2_complete", False)).fillna(False).sum())
    n_af2_pass = int(pd.Series(merged.get("af2_pass_all", False)).fillna(False).sum())
    n_final = int(pd.Series(merged.get("combined_structural_pass", False)).fillna(False).sum())

    stages = [
        ("Context candidate pool", n_all),
        ("Top-N selected", n_topn),
        ("Sequence QC pass", n_seq_qc),
        ("Rosetta scored", n_rosetta_scored),
        ("Rosetta ddG <= 0 REU", n_rosetta_pass),
        ("AF2 completed", n_af2_complete),
        ("AF2 quality+geometry pass", n_af2_pass),
        ("Final combined pass", n_final),
    ]
    base = float(n_all) if n_all > 0 else np.nan
    rows = []
    prev = np.nan
    for stage, n in stages:
        pct_pool = (100.0 * n / base) if np.isfinite(base) and base > 0 else np.nan
        pct_prev = (100.0 * n / prev) if np.isfinite(prev) and prev > 0 else np.nan
        rows.append({"stage": stage, "count": n, "pct_of_pool": pct_pool, "pct_of_previous": pct_prev})
        prev = float(n)
    pd.DataFrame(rows).to_csv(out_dir / "summary_filter_counts_with_pct.csv", index=False)

    metric_spec = [
        ("pred_residual", "Context residual dTm (degC)"),
        ("pred_abs", "Context absolute dTm (degC)"),
        ("surrogate_dtm_wt_centered_cal", "Surrogate WT-centered dTm (degC)"),
        ("ddg_reu", "Rosetta ddG (REU)"),
        ("af2_ranked0_plddt_mean", "AF2 ranked-0 mean pLDDT"),
        ("af2_ranked0_plddt_key_mean", "AF2 ranked-0 key-site mean pLDDT"),
    ]
    mrows = []
    for col, label in metric_spec:
        if col not in merged.columns:
            continue
        s = pd.to_numeric(merged[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        mrows.append(
            {
                "metric": label,
                "column": col,
                "n": int(len(s)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    pd.DataFrame(mrows).to_csv(out_dir / "summary_key_metrics_median_range.csv", index=False)

    if wt_percentiles_csv.exists():
        wt_pct = pd.read_csv(wt_percentiles_csv)
        keep_cols = [
            c
            for c in [
                "metric",
                "direction",
                "wt_value",
                "shortlist_mean",
                "wt_percentile_numeric",
                "wt_percentile_desirability",
                "zscore_vs_shortlist",
            ]
            if c in wt_pct.columns
        ]
        wt_pct[keep_cols].to_csv(out_dir / "summary_wt_percentiles_key_metrics.csv", index=False)

    threshold_rows = [
        ("Sequence filter", "mut_count", "<= 5"),
        ("Sequence filter", "protected_mutations", "== 0"),
        ("Sequence filter", "disulfide_seq_ok", "True"),
        ("Sequence filter", "glyco_motif_new", "False"),
        ("Sequence filter", "duplicate_sequence", "False"),
        ("Sequence filter", "in_training_seq", "False"),
        ("Rosetta filter", "ddg_reu", "<= 0.0"),
        ("AF2 confidence filter", "af2_ranked0_plddt_mean", ">= 90"),
        ("AF2 confidence filter", "af2_ranked0_plddt_key_mean", ">= 90"),
        ("AF2 geometry filter", "af2_triad_ser160_his237 & af2_triad_asp206_his237", "<= 5.0 A"),
        ("AF2 geometry filter", "af2_disulfide_203_239_dist & af2_disulfide_273_289_dist", "<= 2.8 A"),
    ]
    pd.DataFrame(threshold_rows, columns=["stage", "criterion", "threshold"]).to_csv(
        out_dir / "summary_thresholds_used.csv", index=False
    )

    wt_lines = []
    if wt_row is not None:
        wt_lines = [
            f"- WT context residual dTm: {float(wt_row.get('pred_residual', np.nan)):.4f} degC",
            f"- WT context absolute dTm: {float(wt_row.get('pred_abs', np.nan)):.4f} degC",
            f"- WT surrogate WT-centered dTm: {float(wt_row.get('surrogate_dtm_wt_centered_cal', np.nan)):.6f} degC",
            f"- WT Rosetta reference ddG: {float(wt_row.get('ddg_reu_reference', np.nan)):.4f} REU",
            f"- WT AF2 ranked-0 mean pLDDT: {float(wt_row.get('af2_ranked0_plddt_mean', np.nan)):.4f}",
            f"- WT AF2 key-site pLDDT: {float(wt_row.get('af2_ranked0_plddt_key_mean', np.nan)):.4f}",
        ]

    md_lines = [
        "# Final-Shortlist Abstract-Ready Summary",
        "",
        "## Filter Counts",
    ]
    for r in rows:
        pct_pool = "nan" if not np.isfinite(r["pct_of_pool"]) else f"{r['pct_of_pool']:.2f}%"
        md_lines.append(f"- {r['stage']}: n={r['count']} ({pct_pool} of pool)")
    md_lines.extend(
        [
            "",
            "## Key Metrics (Median [Min, Max])",
        ]
    )
    for r in mrows:
        md_lines.append(f"- {r['metric']}: {r['median']:.4f} [{r['min']:.4f}, {r['max']:.4f}]")
    if wt_lines:
        md_lines.extend(["", "## WT Reference Values"])
        md_lines.extend(wt_lines)
    md_lines.extend(
        [
            "",
            "## Thresholds",
        ]
    )
    for stage, criterion, thr in threshold_rows:
        md_lines.append(f"- {stage}: `{criterion}` {thr}")
    (out_dir / "summary_abstract_ready.md").write_text("\n".join(md_lines) + "\n")


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
    merged = prepare_merged_with_surrogate(merged, args.surrogate_csv)
    wt_row = load_wt_row(args.wt_csv)

    plot_selection_landscape(
        merged=merged,
        wt_row=wt_row,
        out_stem=args.out_dir / "fig10_selection_landscape_wt",
        label_top_n=args.label_top_n,
    )

    wt_map = read_fasta(args.wt_fasta)
    wt_seq = next(iter(wt_map.values())) if wt_map else ""
    shortlist_map = read_fasta(args.shortlist_fasta)
    keep_ids = set(merged["id"].astype(str))
    shortlist_map = {k: v for k, v in shortlist_map.items() if k in keep_ids}
    pos_df, sub_df = mutation_stats(wt_seq, shortlist_map)
    pos_df.to_csv(args.out_dir / "summary_mutation_frequency_by_position.csv", index=False)
    sub_df.to_csv(args.out_dir / "summary_recurring_substitutions.csv", index=False)
    plot_mutation_map(
        pos_df=pos_df,
        sub_df=sub_df,
        out_stem=args.out_dir / "fig11_mutation_landscape",
        top_substitutions=args.top_substitutions,
    )

    plot_af2_uncertainty(
        merged=merged,
        wt_row=wt_row,
        out_stem=args.out_dir / "fig12_af2_model_spread_top_candidates",
        top_n=args.uncertainty_top_n,
    )

    write_summary_tables(
        merged=merged,
        wt_row=wt_row,
        wt_percentiles_csv=args.wt_percentiles_csv,
        predictions_csv=args.predictions_csv,
        topn_csv=args.topn_csv,
        filtered_csv=args.filtered_csv,
        rosetta_csv=args.rosetta_csv,
        out_dir=args.out_dir,
    )

    print(f"Wrote deep-dive figures + tables to {args.out_dir}")


if __name__ == "__main__":
    main()
