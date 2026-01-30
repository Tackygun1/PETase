#!/usr/bin/env python3
"""
Plot differences between top-k candidate sequences and WT PETase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib not installed; run: pip install matplotlib") from exc

try:
    from src.scoring.constraints import PROTECTED_POSITIONS
except Exception:
    PROTECTED_POSITIONS = {
        160,
        206,
        237,
        87,
        161,
        185,
        203,
        239,
        273,
        289,
    }

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot top-k sequence differences vs WT.")
    p.add_argument(
        "--topk-fasta",
        type=Path,
        default=Path("results/surrogate_af2/topk.fasta"),
        help="FASTA with top-k candidate sequences.",
    )
    p.add_argument(
        "--wt-fasta",
        type=Path,
        default=Path("data/processed/petase_wt.fasta"),
        help="FASTA with WT PETase sequence.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/plots"),
        help="Output directory for plots.",
    )
    p.add_argument(
        "--top-positions",
        type=int,
        default=20,
        help="Number of most-mutated positions to show in heatmap.",
    )
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


def mutation_matrix(seqs: Sequence[str], wt: str) -> np.ndarray:
    L = len(wt)
    mat = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        mat[i] = np.array([a != b for a, b in zip(s, wt)], dtype=bool)
    return mat


def plot_mutation_counts(ids: Sequence[str], counts: np.ndarray, out: Path) -> None:
    order = np.argsort(-counts)
    sorted_ids = [ids[i] for i in order]
    sorted_counts = counts[order]
    plt.figure(figsize=(max(8, len(ids) * 0.4), 4))
    plt.bar(range(len(sorted_ids)), sorted_counts, color="#3b6fb6")
    plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=90, fontsize=8)
    plt.ylabel("Mutation count vs WT")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_position_frequency(freq: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = np.arange(1, len(freq) + 1)
    plt.plot(x, freq, color="#cf4d3b", linewidth=1.5)
    for pos in sorted(PROTECTED_POSITIONS):
        if 1 <= pos <= len(freq):
            plt.axvline(pos, color="#666666", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.xlabel("Position")
    plt.ylabel("Mutation frequency (top-k)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_substitution_heatmap(
    seqs: Sequence[str],
    wt: str,
    top_positions: int,
    out: Path,
) -> None:
    L = len(wt)
    pos_counts = np.zeros(L, dtype=int)
    for s in seqs:
        pos_counts += np.array([a != b for a, b in zip(s, wt)], dtype=int)
    top_idx = np.argsort(-pos_counts)[: top_positions]
    top_idx = [i for i in top_idx if pos_counts[i] > 0]
    if not top_idx:
        return

    mat = np.zeros((len(AMINO_ACIDS), len(top_idx)), dtype=int)
    for col, i in enumerate(top_idx):
        for s in seqs:
            aa = s[i]
            if aa in AMINO_ACIDS:
                mat[AMINO_ACIDS.index(aa), col] += 1

    plt.figure(figsize=(max(6, len(top_idx) * 0.5), 6))
    plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(label="Count")
    plt.yticks(range(len(AMINO_ACIDS)), AMINO_ACIDS)
    plt.xticks(range(len(top_idx)), [str(i + 1) for i in top_idx], rotation=90)
    plt.xlabel("Position (top mutated)")
    plt.ylabel("Amino acid")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    topk = read_fasta(args.topk_fasta)
    if not topk:
        raise SystemExit(f"No sequences found in {args.topk_fasta}")
    wt_map = read_fasta(args.wt_fasta)
    if not wt_map:
        raise SystemExit(f"No WT sequence found in {args.wt_fasta}")
    wt_seq = next(iter(wt_map.values()))

    ids = list(topk.keys())
    seqs = list(topk.values())
    if any(len(s) != len(wt_seq) for s in seqs):
        raise SystemExit("Sequence lengths do not match WT; cannot compare.")

    mat = mutation_matrix(seqs, wt_seq)
    counts = mat.sum(axis=1)
    freq = mat.mean(axis=0)

    out_dir = args.out_dir
    plot_mutation_counts(ids, counts, out_dir / "mutation_counts.png")
    plot_position_frequency(freq, out_dir / "mutation_frequency.png")
    plot_substitution_heatmap(seqs, wt_seq, args.top_positions, out_dir / "substitution_heatmap.png")

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
