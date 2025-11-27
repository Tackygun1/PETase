"""Generate candidate mutants and export to FASTA for embedding.

IDs follow the proposer convention (e.g., parentid_m0), ensuring embeddings align with run_round.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from acquisition.proposer import propose_mutations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mutant candidates and write FASTA.")
    p.add_argument("--parent-fasta", type=Path, required=True, help="Parent FASTA (first record used).")
    p.add_argument("--candidate-sites", type=str, default=None, help="Comma-separated 1-based positions. Defaults to built-in hotspots.")
    p.add_argument("--max-mutations", type=int, default=1, help="Maximum mutations per candidate (1 or 2).")
    p.add_argument("--output", type=Path, default=Path("data/processed/candidates.fasta"))
    return p.parse_args()


def read_first_fasta(path: Path) -> tuple[str, str]:
    header = None
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is None:
                    header = line[1:].strip()
                else:
                    break
            else:
                seq.append(line)
    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    return header, "".join(seq)


def write_fasta(candidates, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for cand in candidates:
            f.write(f">{cand.seq_id}\n")
            f.write(f"{cand.sequence}\n")


def main() -> None:
    args = parse_args()
    parent_id, parent_seq = read_first_fasta(args.parent_fasta)
    sites: Sequence[int] | None = None
    if args.candidate_sites:
        sites = [int(s.strip()) for s in args.candidate_sites.split(",") if s.strip()]
    candidates = propose_mutations(
        parent_id=parent_id,
        parent_seq=parent_seq,
        candidate_sites=sites,
        max_mutations=args.max_mutations,
    )
    write_fasta(candidates, args.output)
    print(f"Wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
