#!/usr/bin/env python3
"""
Compute consensus ranking using DTM + TM surrogates and generate comparison plots.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib not installed; run: pip install matplotlib") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus ranking using DTM + TM surrogates.")
    p.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/candidates_esm.npz"),
        help="Candidate embeddings NPZ.",
    )
    p.add_argument(
        "--candidates-fasta",
        type=Path,
        default=Path("data/processed/candidates.fasta"),
        help="Candidates FASTA (for top-k output).",
    )
    p.add_argument(
        "--dtm-predictions",
        type=Path,
        default=Path("results/surrogate_af2/predictions_calibrated.csv"),
        help="DTM predictions CSV.",
    )
    p.add_argument(
        "--dtm-col",
        type=str,
        default="prediction_calibrated_wt_centered",
        help="Column name for DTM predictions.",
    )
    p.add_argument(
        "--tm-model",
        type=Path,
        default=Path("models/fireprot_xgb_tm.json"),
        help="TM XGBoost model JSON.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/consensus"),
        help="Output directory for rankings + plots.",
    )
    p.add_argument("--top-k", type=int, default=20, help="Top-k sequences to export.")
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


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std(ddof=1) if values.size > 1 else 1.0
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rank_desc(x).astype(float)
    ry = rank_desc(y).astype(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def plot_scatter(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=18, color="#3b6fb6", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_overlay_hist(a: np.ndarray, b: np.ndarray, out: Path, title: str, label_a: str, label_b: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(a, bins=30, alpha=0.6, label=label_a, color="#cf4d3b")
    plt.hist(b, bins=30, alpha=0.6, label=label_b, color="#3b6fb6")
    plt.title(title)
    plt.xlabel("Z-score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def write_fasta(records: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rid, seq in records:
            f.write(f">{rid}\n{seq}\n")


def main() -> None:
    args = parse_args()

    embeddings = load_embeddings(args.embeddings)
    dtm_preds = load_predictions(args.dtm_predictions, args.dtm_col)
    if not embeddings:
        raise SystemExit(f"No embeddings found in {args.embeddings}")
    if not dtm_preds:
        raise SystemExit(f"No DTM predictions found in {args.dtm_predictions}")

    ids = [rid for rid in embeddings.keys() if rid in dtm_preds]
    if not ids:
        raise SystemExit("No overlapping IDs between embeddings and DTM predictions.")

    X = np.vstack([embeddings[rid] for rid in ids])
    dtm_vals = np.array([dtm_preds[rid] for rid in ids], dtype=float)

    import xgboost as xgb

    tm_model = xgb.XGBRegressor()
    tm_model.load_model(args.tm_model)
    tm_vals = tm_model.predict(X)

    z_dtm = zscore(dtm_vals)
    z_tm = zscore(tm_vals)
    z_consensus = (z_dtm + z_tm) / 2.0

    rank_dtm = rank_desc(z_dtm)
    rank_tm = rank_desc(z_tm)
    rank_cons = rank_desc(z_consensus)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write consensus table
    out_csv = out_dir / "predictions_consensus.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "dtm_value",
                "tm_value",
                "z_dtm",
                "z_tm",
                "z_consensus",
                "rank_dtm",
                "rank_tm",
                "rank_consensus",
            ]
        )
        for i, rid in enumerate(ids):
            writer.writerow(
                [
                    rid,
                    f"{dtm_vals[i]:.6f}",
                    f"{tm_vals[i]:.6f}",
                    f"{z_dtm[i]:.6f}",
                    f"{z_tm[i]:.6f}",
                    f"{z_consensus[i]:.6f}",
                    int(rank_dtm[i]),
                    int(rank_tm[i]),
                    int(rank_cons[i]),
                ]
            )

    # Plots: DTM vs TM values and rank scatter
    rho_pred = spearman(dtm_vals, tm_vals)
    rho_rank = spearman(-rank_dtm.astype(float), -rank_tm.astype(float))

    plot_scatter(
        dtm_vals,
        tm_vals,
        out_dir / "scatter_dtm_vs_tm.png",
        f"DTM vs TM predictions (Spearman={rho_pred:.2f})",
        "DTM (WT-centered)",
        "TM",
    )
    plot_scatter(
        rank_dtm.astype(float),
        rank_tm.astype(float),
        out_dir / "scatter_rank_dtm_vs_tm.png",
        f"Rank comparison (Spearman={rho_rank:.2f})",
        "Rank by DTM",
        "Rank by TM",
    )
    plot_overlay_hist(
        z_dtm,
        z_tm,
        out_dir / "zscore_distributions.png",
        "Z-score distributions (DTM vs TM)",
        "DTM",
        "TM",
    )

    # Write top-k consensus FASTA
    fasta_map = read_fasta(args.candidates_fasta)
    if fasta_map:
        order = np.argsort(-z_consensus)
        top_ids = [ids[i] for i in order[: args.top_k]]
        records = [(rid, fasta_map[rid]) for rid in top_ids if rid in fasta_map]
        write_fasta(records, out_dir / "topk_consensus.fasta")

    # Summary
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["spearman_pred", f"{rho_pred:.6f}"])
        writer.writerow(["spearman_rank", f"{rho_rank:.6f}"])
        writer.writerow(["n", str(len(ids))])

    print(f"Wrote consensus outputs to {out_dir}")


if __name__ == "__main__":
    main()
