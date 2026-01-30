#!/usr/bin/env python3
"""
Statistical tests + plots comparing top-k candidates vs background.

Tests:
- Permutation tests for predicted stability and mutation counts
- Diversity test (mean pairwise Hamming distance) vs random sets
- Positional enrichment (Fisher exact) with frequency plots
- Motif enrichment (N-X-S/T new motif)
- Spearman correlation between prediction and pLDDT (if AF2 metrics exist)
- Z-score outlier view for top-k predictions
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib not installed; run: pip install matplotlib") from exc


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-k statistical analysis + plots.")
    p.add_argument(
        "--topk-fasta",
        type=Path,
        default=Path("results/surrogate_af2/topk.fasta"),
        help="FASTA with top-k sequences.",
    )
    p.add_argument(
        "--candidates-fasta",
        type=Path,
        default=Path("data/processed/candidates.fasta"),
        help="FASTA with all candidates.",
    )
    p.add_argument(
        "--wt-fasta",
        type=Path,
        default=Path("data/processed/petase_wt.fasta"),
        help="FASTA with WT PETase sequence.",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/predictions.csv"),
        help="Surrogate predictions CSV (id,prediction).",
    )
    p.add_argument(
        "--prediction-col",
        type=str,
        default="prediction",
        help="Prediction column name in --predictions.",
    )
    p.add_argument(
        "--predictions-with-af2",
        type=Path,
        default=Path("results/surrogate_af2/predictions_with_af2.csv"),
        help="Merged predictions + AF2 metrics CSV (optional).",
    )
    p.add_argument(
        "--qc-report",
        type=Path,
        default=Path("results/surrogate_af2/qc_report.csv"),
        help="QC report CSV with motif flags (optional).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/stats_topk"),
        help="Output directory for plots and summary.",
    )
    p.add_argument("--n-perm", type=int, default=2000, help="Permutation iterations.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--top-positions", type=int, default=20, help="Top enriched positions to plot.")
    return p.parse_args()


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
                if current_id:
                    seqs[current_id] = "".join(parts)
                current_id = line[1:].strip().split()[0]
                parts = []
            else:
                parts.append(line)
        if current_id:
            seqs[current_id] = "".join(parts)
    return seqs


def load_predictions(path: Path, col: str) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    if not path.exists():
        return preds
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return preds
        if "id" not in reader.fieldnames or col not in reader.fieldnames:
            return preds
        for row in reader:
            rid = (row.get("id") or "").strip()
            val = (row.get(col) or "").strip()
            if not rid or not val:
                continue
            try:
                preds[rid] = float(val)
            except ValueError:
                continue
    return preds


def load_af2_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return metrics
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return metrics
        needed = {"id", "plddt_mean", "plddt_min"}
        if not needed.issubset(set(reader.fieldnames)):
            return metrics
        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            try:
                metrics[rid] = {
                    "plddt_mean": float(row.get("plddt_mean", "nan")),
                    "plddt_min": float(row.get("plddt_min", "nan")),
                }
            except ValueError:
                continue
    return metrics


def glyco_motif_count(seq: str) -> int:
    if len(seq) < 3:
        return 0
    count = 0
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 2] in ("S", "T"):
            count += 1
    return count


def load_qc_motif(path: Path) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    if not path.exists():
        return flags
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            return flags
        key = "glyco_motif_new"
        if key not in reader.fieldnames:
            return flags
        for row in reader:
            rid = (row.get("id") or "").strip()
            val = (row.get(key) or "").strip()
            if rid:
                flags[rid] = val.lower() == "true"
    return flags


def encode_sequences(seqs: Sequence[str]) -> np.ndarray:
    aa_map = {aa: i for i, aa in enumerate(AMINO_ACIDS, start=1)}
    arr = np.zeros((len(seqs), len(seqs[0])), dtype=np.uint8)
    for i, s in enumerate(seqs):
        arr[i] = np.array([aa_map.get(ch, 0) for ch in s], dtype=np.uint8)
    return arr


def mutation_matrix(arr: np.ndarray, wt_arr: np.ndarray) -> np.ndarray:
    return arr != wt_arr


def mean_pairwise_hamming(arr: np.ndarray) -> float:
    k = arr.shape[0]
    if k < 2:
        return 0.0
    diffs = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    idx = np.triu_indices(k, 1)
    return float(diffs[idx].mean())


def permutation_test_mean_diff(values: np.ndarray, top_idx: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float, np.ndarray]:
    top_vals = values[top_idx]
    bg_vals = values[~top_idx]
    obs = float(top_vals.mean() - bg_vals.mean())
    n_top = top_vals.size
    perm_diffs = []
    for _ in range(n_perm):
        perm_idx = rng.choice(values.size, size=n_top, replace=False)
        perm_top = values[perm_idx]
        perm_bg = np.delete(values, perm_idx)
        perm_diffs.append(float(perm_top.mean() - perm_bg.mean()))
    perm_diffs = np.array(perm_diffs)
    p_one = float((np.sum(perm_diffs >= obs) + 1) / (n_perm + 1))
    return obs, p_one, perm_diffs


def rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value."""
    def log_choose(n: int, k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    n = a + b + c + d
    r1 = a + b
    c1 = a + c
    def log_hyper(x: int) -> float:
        return log_choose(c1, x) + log_choose(n - c1, r1 - x) - log_choose(n, r1)

    obs = log_hyper(a)
    lo = max(0, r1 - (n - c1))
    hi = min(r1, c1)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = log_hyper(x)
        if lp <= obs + 1e-12:
            total += math.exp(lp)
    return min(1.0, total)


def plot_hist_with_line(values: np.ndarray, obs: float, out: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color="#8fb3d9", edgecolor="white")
    plt.axvline(obs, color="#c43c39", linewidth=2, label=f"Observed={obs:.3f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_box(top: np.ndarray, bg: np.ndarray, out: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.boxplot([top, bg], labels=["Top-k", "Background"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_line(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, out: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label="Top-k", color="#cf4d3b", linewidth=1.5)
    plt.plot(x, y2, label="Background", color="#3b6fb6", linewidth=1.2)
    plt.xlabel("Position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_bar(labels: Sequence[str], values: Sequence[float], out: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(max(6, len(labels) * 0.4), 4))
    plt.bar(range(len(labels)), values, color="#6d9f71")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_scatter(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=30, color="#3b6fb6")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    topk_map = read_fasta(args.topk_fasta)
    cand_map = read_fasta(args.candidates_fasta)
    wt_map = read_fasta(args.wt_fasta)
    if not topk_map:
        raise SystemExit(f"No sequences found in {args.topk_fasta}")
    if not cand_map:
        raise SystemExit(f"No sequences found in {args.candidates_fasta}")
    if not wt_map:
        raise SystemExit(f"No WT sequence found in {args.wt_fasta}")

    wt_seq = next(iter(wt_map.values()))
    topk_ids = list(topk_map.keys())
    cand_ids = list(cand_map.keys())

    preds = load_predictions(args.predictions, args.prediction_col)
    topk_preds = np.array([preds.get(i) for i in topk_ids], dtype=float)
    cand_preds = np.array([preds.get(i) for i in cand_ids], dtype=float)

    valid_pred_mask = ~np.isnan(cand_preds)
    cand_ids = [cid for cid, ok in zip(cand_ids, valid_pred_mask) if ok]
    cand_preds = cand_preds[valid_pred_mask]
    in_map = [cid in cand_map for cid in cand_ids]
    cand_ids = [cid for cid, ok in zip(cand_ids, in_map) if ok]
    cand_preds = cand_preds[in_map]

    topk_pred_mask = ~np.isnan(topk_preds)
    topk_ids = [cid for cid, ok in zip(topk_ids, topk_pred_mask) if ok]
    topk_preds = topk_preds[topk_pred_mask]

    if len(topk_ids) == 0:
        raise SystemExit("No top-k predictions found; check predictions.csv.")

    # Encode sequences and mutation matrix
    cand_seqs = [cand_map[i] for i in cand_ids]
    topk_seqs = [topk_map[i] for i in topk_ids]
    if any(len(s) != len(wt_seq) for s in cand_seqs + topk_seqs):
        raise SystemExit("Sequence lengths do not match WT.")

    cand_arr = encode_sequences(cand_seqs)
    topk_arr = encode_sequences(topk_seqs)
    wt_arr = encode_sequences([wt_seq])[0]
    mut_mat = mutation_matrix(cand_arr, wt_arr)
    topk_mut_mat = mutation_matrix(topk_arr, wt_arr)
    cand_mut_counts = mut_mat.sum(axis=1).astype(float)
    topk_mut_counts = topk_mut_mat.sum(axis=1).astype(float)

    # Permutation tests: prediction + mutation count
    top_idx = np.array([cid in set(topk_ids) for cid in cand_ids], dtype=bool)
    pred_obs, pred_p, pred_perm = permutation_test_mean_diff(cand_preds, top_idx, args.n_perm, rng)
    mut_obs, mut_p, mut_perm = permutation_test_mean_diff(cand_mut_counts, top_idx, args.n_perm, rng)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hist_with_line(
        pred_perm,
        pred_obs,
        out_dir / "perm_pred_diff.png",
        "Permutation test: prediction mean diff (top - bg)",
        "Mean diff",
    )
    plot_box(topk_preds, cand_preds[~top_idx], out_dir / "prediction_box.png", "Predictions", "Predicted DTM")

    plot_hist_with_line(
        mut_perm,
        mut_obs,
        out_dir / "perm_mutcount_diff.png",
        "Permutation test: mutation count mean diff (top - bg)",
        "Mean diff",
    )
    plot_box(topk_mut_counts, cand_mut_counts[~top_idx], out_dir / "mutation_count_box.png", "Mutation counts", "Mutations vs WT")

    # Diversity test
    topk_div = mean_pairwise_hamming(topk_arr)
    div_perm = []
    for _ in range(args.n_perm):
        sample_idx = rng.choice(cand_arr.shape[0], size=len(topk_arr), replace=False)
        div_perm.append(mean_pairwise_hamming(cand_arr[sample_idx]))
    div_perm = np.array(div_perm)
    div_p = float((np.sum(div_perm >= topk_div) + 1) / (args.n_perm + 1))
    plot_hist_with_line(
        div_perm,
        topk_div,
        out_dir / "perm_diversity.png",
        "Diversity vs random sets (mean pairwise Hamming)",
        "Mean pairwise Hamming",
    )

    # Positional enrichment
    freq_top = topk_mut_mat.mean(axis=0)
    freq_bg = mut_mat.mean(axis=0)
    x = np.arange(1, len(wt_seq) + 1)
    plot_line(x, freq_top, freq_bg, out_dir / "mutation_frequency.png", "Mutation frequency by position", "Frequency")

    enrich = []
    for i in range(len(wt_seq)):
        a = int(topk_mut_mat[:, i].sum())  # top mutated
        b = int(len(topk_arr) - a)        # top not mutated
        c = int(mut_mat[:, i].sum())      # bg mutated
        d = int(len(mut_mat) - c)         # bg not mutated
        p = fisher_exact(a, b, c, d)
        delta = float(freq_top[i] - freq_bg[i])
        enrich.append((i + 1, p, delta))
    enrich.sort(key=lambda t: t[1])
    top_enrich = [e for e in enrich if e[2] > 0][: args.top_positions]
    if top_enrich:
        labels = [str(e[0]) for e in top_enrich]
        values = [-math.log10(e[1]) if e[1] > 0 else 50.0 for e in top_enrich]
        plot_bar(labels, values, out_dir / "position_enrichment.png", "Top enriched positions (-log10 p)", "-log10(p)")

    # Motif enrichment (new N-X-S/T)
    motif_flags = load_qc_motif(args.qc_report)
    if motif_flags:
        top_motif = sum(1 for i in topk_ids if motif_flags.get(i, False))
        bg_motif = sum(1 for i in cand_ids if motif_flags.get(i, False))
    else:
        wt_motifs = glyco_motif_count(wt_seq)
        top_motif = sum(1 for s in topk_seqs if glyco_motif_count(s) > wt_motifs)
        bg_motif = sum(1 for s in cand_seqs if glyco_motif_count(s) > wt_motifs)
    a, b = top_motif, len(topk_seqs) - top_motif
    c, d = bg_motif, len(cand_seqs) - bg_motif
    motif_p = fisher_exact(a, b, c, d)
    plot_bar(
        ["Top-k", "Background"],
        [top_motif / len(topk_seqs), bg_motif / len(cand_seqs)],
        out_dir / "glyco_motif_rate.png",
        "New N-X-S/T motif rate",
        "Rate",
    )

    # Spearman correlation: prediction vs pLDDT (if AF2 metrics exist)
    af2 = load_af2_metrics(args.predictions_with_af2)
    if af2:
        xs = []
        ys = []
        ys_min = []
        for cid in topk_ids:
            pred = preds.get(cid)
            if cid in af2 and pred is not None:
                xs.append(pred)
                ys.append(af2[cid]["plddt_mean"])
                ys_min.append(af2[cid]["plddt_min"])
        if xs:
            xs = np.array(xs)
            ys = np.array(ys)
            ys_min = np.array(ys_min)
            rho_mean = spearman(xs, ys)
            rho_min = spearman(xs, ys_min)
            plot_scatter(xs, ys, out_dir / "pred_vs_plddt_mean.png", f"Spearman r={rho_mean:.2f}", "Prediction", "pLDDT mean")
            plot_scatter(xs, ys_min, out_dir / "pred_vs_plddt_min.png", f"Spearman r={rho_min:.2f}", "Prediction", "pLDDT min")

    # Z-score view
    pred_mean = cand_preds.mean()
    pred_std = cand_preds.std(ddof=1) if cand_preds.size > 1 else 1.0
    z_top = (topk_preds - pred_mean) / pred_std
    plot_bar(topk_ids, z_top.tolist(), out_dir / "prediction_zscores.png", "Top-k prediction z-scores", "Z-score")

    # Summary
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "statistic", "value"])
        writer.writerow(["prediction_mean_diff", "obs", f"{pred_obs:.4f}"])
        writer.writerow(["prediction_mean_diff", "perm_p", f"{pred_p:.4g}"])
        writer.writerow(["mutation_count_mean_diff", "obs", f"{mut_obs:.4f}"])
        writer.writerow(["mutation_count_mean_diff", "perm_p", f"{mut_p:.4g}"])
        writer.writerow(["diversity_mean_pairwise", "obs", f"{topk_div:.4f}"])
        writer.writerow(["diversity_mean_pairwise", "perm_p", f"{div_p:.4g}"])
        writer.writerow(["glyco_motif_new_rate_top", "rate", f"{top_motif/len(topk_seqs):.4f}"])
        writer.writerow(["glyco_motif_new_rate_bg", "rate", f"{bg_motif/len(cand_seqs):.4f}"])
        writer.writerow(["glyco_motif_fisher_p", "p", f"{motif_p:.4g}"])
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
