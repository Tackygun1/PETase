#!/usr/bin/env python3
"""
Generate random single-point mutants from a parent FASTA sequence.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Set, Tuple


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate random single-point mutants.")
    p.add_argument(
        "--parent-fasta",
        type=Path,
        required=True,
        help="FASTA file containing the parent (WT) sequence.",
    )
    p.add_argument("--n", type=int, default=2000, help="Number of mutants to generate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--exclude-positions",
        default="",
        help="Comma-separated 1-based positions to exclude from mutation.",
    )
    p.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Optional file with one 1-based position per line to exclude.",
    )
    p.add_argument(
        "--out-fasta",
        type=Path,
        default=Path("data/processed/guardrail_random_mutants.fasta"),
        help="Output FASTA path.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/guardrail_random_mutants.csv"),
        help="Output CSV path with id/sequence/mutation info.",
    )
    return p.parse_args()


def read_first_fasta(path: Path) -> Tuple[str, str]:
    seq_id = None
    seq_parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is None:
                    seq_id = line[1:].strip()
                else:
                    break
            else:
                seq_parts.append(line)
    if seq_id is None or not seq_parts:
        raise ValueError(f"No FASTA sequence found in {path}")
    return seq_id, "".join(seq_parts)


def parse_excludes(args: argparse.Namespace) -> Set[int]:
    excludes: Set[int] = set()
    if args.exclude_positions:
        for tok in args.exclude_positions.split(","):
            tok = tok.strip()
            if tok:
                excludes.add(int(tok))
    if args.exclude_file and args.exclude_file.exists():
        for line in args.exclude_file.read_text().splitlines():
            line = line.strip()
            if line:
                excludes.add(int(line))
    return excludes


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    parent_id, parent_seq = read_first_fasta(args.parent_fasta)
    excludes = parse_excludes(args)

    allowed_positions = [i for i in range(1, len(parent_seq) + 1) if i not in excludes]
    if not allowed_positions:
        raise SystemExit("No positions left to mutate after exclusions.")

    rows = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = args.n * 20
    while len(rows) < args.n and attempts < max_attempts:
        attempts += 1
        pos = rng.choice(allowed_positions)
        ref = parent_seq[pos - 1]
        alt = rng.choice([aa for aa in AA_LIST if aa != ref])
        mut_seq = parent_seq[: pos - 1] + alt + parent_seq[pos:]
        if mut_seq in seen:
            continue
        seen.add(mut_seq)
        mut_id = f"{parent_id}_rand_{ref}{pos}{alt}"
        rows.append(
            {
                "id": mut_id,
                "sequence": mut_seq,
                "mutation": f"{ref}{pos}{alt}",
                "position": pos,
            }
        )

    if len(rows) < args.n:
        print(f"Warning: generated {len(rows)} unique mutants (requested {args.n}).")

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_fasta, "w") as f:
        for row in rows:
            f.write(f">{row['id']}\n")
            f.write(f"{row['sequence']}\n")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "mutation", "position"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out_fasta} and {args.out_csv} with {len(rows)} mutants")


if __name__ == "__main__":
    main()
