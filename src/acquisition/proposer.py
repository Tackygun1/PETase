"""Mutation proposer that respects hard structural constraints."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

from .acquisition import Candidate, apply_mutations
from ..scoring.constraints import is_allowed_position
from ..utils.io import load_labels_csv

# Conservative amino acid set for single-site substitutions.
DEFAULT_MUT_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Design hotspots drawn from the PDFs (loops, Arg280 cap, known stabilizers).
# 121 (loop rigidifying), 181 (β6 strand), 224/233/280 (FAST-PETase positions), 95/201 (stability tradeoffs),
# 229/274 (near surface), 84/88 (loop around active site), 260 (surface), 96/193 (packing), 214 (loop),
# 238 (loop near disulfide), 185 (kept but allowed for conservative scans), 208 (subsite I neighbor).
DEFAULT_HOTSPOT_SITES = [
    84,
    88,
    95,
    96,
    121,
    181,
    185,
    193,
    201,
    208,
    214,
    224,
    229,
    233,
    238,
    260,
    274,
    280,
]

# Helper to extract mutations from neighbor sequences.
def mutations_from_neighbor(parent_seq: str, neighbor_seq: str) -> List[str]:
    muts: List[str] = []
    for i, (a, b) in enumerate(zip(parent_seq, neighbor_seq), start=1):
        if a != b:
            muts.append(_mutation_string(a, i, b))
    return muts


def _mutation_string(orig: str, pos: int, dest: str) -> str:
    return f"{orig}{pos}{dest}"


def propose_mutations(
    parent_id: str,
    parent_seq: str,
    candidate_sites: Iterable[int] | None = None,
    max_mutations: int = 1,
    allowed_aas: Sequence[str] = DEFAULT_MUT_AMINO_ACIDS,
) -> List[Candidate]:
    """Generate candidate mutations, filtering out protected sites.

    Returns a list of Candidate objects with sequences and mutation lists populated. Predicted scores
    remain unset for downstream scoring.
    """
    parent_seq = parent_seq.strip().upper()
    seq_chars = list(parent_seq)
    if candidate_sites is None:
        candidate_sites = DEFAULT_HOTSPOT_SITES
    valid_sites = [s for s in candidate_sites if is_allowed_position(s)]

    mutations: List[List[str]] = []

    # Single mutations
    for pos in valid_sites:
        orig = seq_chars[pos - 1]
        for aa in allowed_aas:
            if aa == orig:
                continue
            mutations.append([_mutation_string(orig, pos, aa)])

    # Optional double mutants (simple combinations)
    if max_mutations >= 2:
        for (pos1, pos2) in itertools.combinations(valid_sites, 2):
            orig1, orig2 = seq_chars[pos1 - 1], seq_chars[pos2 - 1]
            for aa1 in allowed_aas:
                if aa1 == orig1:
                    continue
                for aa2 in allowed_aas:
                    if aa2 == orig2:
                        continue
                    mutations.append(
                        [
                            _mutation_string(orig1, pos1, aa1),
                            _mutation_string(orig2, pos2, aa2),
                        ]
                    )

    candidates: List[Candidate] = []
    for idx, muts in enumerate(mutations):
        seq = apply_mutations(parent_seq, muts)
        cid = f"{parent_id}_m{idx}"
        candidates.append(
            Candidate(
                seq_id=cid,
                sequence=seq,
                mutations=muts,
                mut_count=len(muts),
                pred_stab_mean=0.0,
                pred_stab_std=0.0,
            )
        )
    return candidates


def propose_from_neighbors(
    parent_id: str,
    parent_seq: str,
    neighbors: List[str],
    neighbor_ids: List[str],
    max_mutations: int = 2,
    candidate_limit: int = 100,
) -> List[Candidate]:
    """Generate candidates by harvesting mutations present in retrieved neighbor sequences."""
    parent_seq = parent_seq.strip().upper()
    candidates: List[Candidate] = []
    for nid, nseq in zip(neighbor_ids, neighbors):
        muts = mutations_from_neighbor(parent_seq, nseq.upper())
        # Filter out protected sites
        muts = [m for m in muts if is_allowed_position(int(m[1:-1]))]
        if not muts:
            continue
        # Limit to max_mutations simple subsets
        muts = muts[:max_mutations]
        seq = apply_mutations(parent_seq, muts)
        cid = f"{parent_id}_rag_{nid}"
        candidates.append(
            Candidate(
                seq_id=cid,
                sequence=seq,
                mutations=muts,
                mut_count=len(muts),
                pred_stab_mean=0.0,
                pred_stab_std=0.0,
            )
        )
        if len(candidates) >= candidate_limit:
            break
    return candidates
