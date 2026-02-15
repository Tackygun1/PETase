#!/usr/bin/env python3
"""
Prepare Rosetta mutfiles (ddg_monomer/cartesian_ddg) from candidate FASTA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Rosetta mutfiles for candidates.")
    p.add_argument(
        "--candidates-fasta",
        type=Path,
        required=True,
        help="FASTA with candidate sequences.",
    )
    p.add_argument(
        "--wt-fasta",
        type=Path,
        required=True,
        help="FASTA with WT sequence (first record used).",
    )
    p.add_argument(
        "--chain",
        default="A",
        help="Chain ID to use in Rosetta mutfiles (default: A).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/rosetta/mutfiles"),
        help="Output directory for mutfiles.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/rosetta/mutations.csv"),
        help="CSV mapping id to mutation string.",
    )
    return p.parse_args()


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
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


def write_mutfile(path: Path, chain: str, muts: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"total {len(muts)}\n")
        f.write(f"{len(muts)}\n")
        for mut in muts:
            ref = mut[0]
            pos = mut[1:-1]
            alt = mut[-1]
            f.write(f"{ref} {pos} {alt}\n")


def main() -> None:
    args = parse_args()
    wt = read_fasta(args.wt_fasta)
    if not wt:
        raise SystemExit("No WT sequence found.")
    wt_seq = next(iter(wt.values()))

    cands = read_fasta(args.candidates_fasta)
    if not cands:
        raise SystemExit("No candidate sequences found.")

    rows = []
    for cid, seq in cands.items():
        muts = mutation_list(wt_seq, seq)
        if not muts:
            continue
        mutfile = args.out_dir / f"{cid}.mutfile"
        write_mutfile(mutfile, args.chain, muts)
        rows.append({"id": cid, "mutation": ";".join(muts), "mutfile": str(mutfile)})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mutation", "mutfile"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} mutfiles to {args.out_dir}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
