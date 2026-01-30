#!/usr/bin/env python3
"""
Prepare mutation lists for DynaMut2 (WT -> mutant).

Outputs a CSV with id, mutation_list, n_mutations and per-variant txt files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DynaMut2 mutation lists from top-k.")
    p.add_argument(
        "--wt-fasta",
        type=Path,
        default=Path("data/processed/petase_wt.fasta"),
        help="WT FASTA (single record).",
    )
    p.add_argument(
        "--topk-fasta",
        type=Path,
        default=Path("results/surrogate_af2/consensus/topk_consensus.fasta"),
        help="Top-k FASTA to compare against WT.",
    )
    p.add_argument(
        "--fallback-topk",
        type=Path,
        default=Path("results/surrogate_af2/topk.fasta"),
        help="Fallback top-k FASTA if --topk-fasta not found.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/surrogate_af2/validation/dynamut2_mutations.csv"),
        help="Output CSV with mutation lists.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/surrogate_af2/validation/dynamut2_mutations"),
        help="Directory for per-variant mutation list files.",
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


def mutation_list(wt: str, seq: str) -> List[str]:
    muts = []
    for i, (a, b) in enumerate(zip(wt, seq), start=1):
        if a != b:
            muts.append(f"{a}{i}{b}")
    return muts


def main() -> None:
    args = parse_args()
    wt_map = read_fasta(args.wt_fasta)
    if not wt_map:
        raise SystemExit(f"No WT sequence found in {args.wt_fasta}")
    wt_seq = next(iter(wt_map.values()))

    topk_map = read_fasta(args.topk_fasta)
    if not topk_map:
        topk_map = read_fasta(args.fallback_topk)
    if not topk_map:
        raise SystemExit("No top-k sequences found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mutation_list", "n_mutations"])
        for sid, seq in topk_map.items():
            if len(seq) != len(wt_seq):
                raise SystemExit(f"Length mismatch for {sid} (WT={len(wt_seq)}).")
            muts = mutation_list(wt_seq, seq)
            mut_str = ";".join(muts)
            writer.writerow([sid, mut_str, len(muts)])
            (args.out_dir / f"{sid}.txt").write_text(mut_str + "\n")

    print(f"Wrote mutation lists to {args.out} and {args.out_dir}")


if __name__ == "__main__":
    main()
