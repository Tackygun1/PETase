"""Generate candidate mutants and export to FASTA for embedding.

IDs follow the proposer convention (e.g., parentid_m0), ensuring embeddings align with run_round.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.acquisition.proposer import propose_mutations
from src.acquisition.batch_design import MutationModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mutant candidates and write FASTA.")
    p.add_argument("--parent-fasta", type=Path, required=True, help="Parent FASTA (first record used).")
    p.add_argument(
        "--mode",
        choices=["hotspot", "retrieval"],
        default="hotspot",
        help="Candidate generator mode (default: hotspot).",
    )
    p.add_argument(
        "--candidate-sites",
        type=str,
        default=None,
        help="Comma-separated 1-based positions (hotspot mode). Defaults to built-in hotspots.",
    )
    p.add_argument(
        "--max-mutations",
        type=int,
        default=1,
        help="Maximum mutations per candidate (hotspot mode).",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embeddings NPZ for retrieval mode.",
    )
    p.add_argument(
        "--sequence-db",
        type=Path,
        default=None,
        help="Sequence DB CSV (id,sequence) for retrieval mode.",
    )
    p.add_argument("--proposals", type=int, default=256, help="Number of proposals to generate.")
    p.add_argument("--trust-min", type=int, default=1, help="Min mutations from neighbor.")
    p.add_argument("--trust-max", type=int, default=5, help="Max mutations from neighbor.")
    p.add_argument(
        "--proposals-per-neighbor",
        type=int,
        default=2,
        help="How many variants to sample per neighbor (retrieval mode).",
    )
    p.add_argument(
        "--proposal-mode",
        choices=["neighbor", "pool"],
        default="pool",
        help="Retrieval proposal strategy (default: pool).",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for retrieval mode.")
    p.add_argument(
        "--extra-protected",
        type=str,
        default=None,
        help="Comma-separated 1-based positions to protect from mutation.",
    )
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
    extra_protected = None
    if args.extra_protected:
        extra_protected = [int(s.strip()) for s in args.extra_protected.split(",") if s.strip()]

    if args.mode == "hotspot":
        sites: Sequence[int] | None = None
        if args.candidate_sites:
            sites = [int(s.strip()) for s in args.candidate_sites.split(",") if s.strip()]
        candidates = propose_mutations(
            parent_id=parent_id,
            parent_seq=parent_seq,
            candidate_sites=sites,
            max_mutations=args.max_mutations,
            extra_protected=extra_protected,
        )
    else:
        if args.embeddings is None or args.sequence_db is None:
            raise ValueError("--embeddings and --sequence-db are required for retrieval mode.")
        mut_model = MutationModel(
            embedding_path=args.embeddings,
            sequence_db=args.sequence_db,
            wt_id=parent_id,
            trust_radius=(args.trust_min, args.trust_max),
            extra_protected=extra_protected,
            seed=args.seed,
        )
        candidates = mut_model.propose(
            k=args.proposals,
            proposals_per_neighbor=args.proposals_per_neighbor,
            proposal_mode=args.proposal_mode,
            seed=args.seed,
        )
    write_fasta(candidates, args.output)
    if args.mode == "retrieval" and len(candidates) < args.proposals:
        print(
            f"Warning: only {len(candidates)} candidates generated from retrieval pool "
            f"(requested {args.proposals})."
        )
    print(f"Wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
