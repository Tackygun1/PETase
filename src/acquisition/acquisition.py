"""
Acquisition functions and constraint handling for PETase variant selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# -------- Constraints --------
@dataclass(frozen=True)
class HardConstraints:
    catalytic_positions: Dict[int, str]
    disulfide_pairs: Sequence[Tuple[int, int]]
    forbid_motif: Sequence[str] = ()
    max_mutations: int = 8

    def violated(self, seq: str, wt_seq: str, mutation_count: int) -> bool:
        if mutation_count > self.max_mutations:
            return True
        # catalytic triad / key residues must match wild-type amino acid
        for pos, aa in self.catalytic_positions.items():
            if seq[pos - 1] != aa:
                return True
        # disulfides: keep cysteines
        for i, j in self.disulfide_pairs:
            if i - 1 < len(seq) and j - 1 < len(seq):
                if seq[i - 1] != "C" or seq[j - 1] != "C":
                    return True
        motif_str = "".join(self.forbid_motif)
        if motif_str and motif_str in seq:
            return True
        return False


@dataclass
class Candidate:
    seq_id: str
    sequence: str
    mutations: List[str]
    mut_count: int
    pred_stab_mean: float = 0.0
    pred_stab_std: float = 0.0
    pred_act_mean: float = 0.0
    acquisition: float = 0.0


def apply_mutations(seq: str, muts: List[str]) -> str:
    """Apply mutation strings like 'S121E' to a sequence with validation."""
    seq_list = list(seq)
    for mut in muts:
        orig, pos, dest = mut[0], int(mut[1:-1]), mut[-1]
        if pos < 1 or pos > len(seq_list):
            raise ValueError(f"Position {pos} out of bounds for sequence length {len(seq_list)}")
        if seq_list[pos - 1] != orig:
            raise ValueError(f"Expected {orig} at position {pos}, found {seq_list[pos - 1]}")
        seq_list[pos - 1] = dest
    return "".join(seq_list)


def filter_by_distance(cands: List[Candidate], min_hamming: int) -> List[Candidate]:
    """Greedy Hamming-distance diversity filter."""
    kept: List[Candidate] = []
    for c in cands:
        if not kept:
            kept.append(c)
            continue
        if all(sum(a != b for a, b in zip(c.sequence, k.sequence)) >= min_hamming for k in kept):
            kept.append(c)
    return kept


def mutation_distance(seq: str, wt: str) -> int:
    return sum(a != b for a, b in zip(seq, wt))


# -------- Acquisition scores --------
def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return mean + beta * std


def thompson_sample(mean: np.ndarray, std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mean, scale=std)


def composite_objective(
    stability: np.ndarray,
    activity: np.ndarray,
    weights: Tuple[float, float] = (1.0, 1.0),
    penalty: float = 1e3,
    constraint_mask: np.ndarray | None = None,
) -> np.ndarray:
    score = weights[0] * stability + weights[1] * activity
    if constraint_mask is not None:
        score = score - penalty * (~constraint_mask)
    return score


def rank_candidates(
    mean: np.ndarray,
    std: np.ndarray,
    wt_seq: str,
    sequences: Sequence[str],
    constraints: HardConstraints,
    weights: Tuple[float, float] = (1.0, 0.5),
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> List[int]:
    """
    Rank candidate indices using UCB + constraints + composite stability/activity score.
    """
    rng = rng or np.random.default_rng()
    stability = mean[:, 0]
    activity = mean[:, 1] if mean.shape[1] > 1 else np.zeros_like(stability)
    stdev = std[:, 0]

    mut_counts = np.array([mutation_distance(s, wt_seq) for s in sequences])
    allowed = np.array(
        [
            not constraints.violated(seq=s, wt_seq=wt_seq, mutation_count=mc)
            for s, mc in zip(sequences, mut_counts)
        ]
    )

    ucb = upper_confidence_bound(stability, stdev, beta=beta)
    comp = composite_objective(stability, activity, weights=weights, constraint_mask=allowed)

    # combine: prioritize constraint satisfaction, then composite, then UCB for exploration
    priority = np.stack([allowed.astype(float), comp, ucb], axis=1)
    order = np.lexsort((-priority[:, 2], -priority[:, 1], -priority[:, 0]))
    return order.tolist()[::-1]


def compute_acquisition(
    candidates: List[Candidate],
    beta: float,
    w_stability: float,
    w_activity: float,
    activity_floor: float | None = None,
) -> List[Candidate]:
    """Assign acquisition scores to Candidate objects in-place."""
    for c in candidates:
        if activity_floor is not None and c.pred_act_mean < activity_floor:
            c.acquisition = -np.inf
            continue
        ucb = c.pred_stab_mean + beta * c.pred_stab_std
        c.acquisition = w_stability * ucb + w_activity * c.pred_act_mean
    return candidates


__all__ = [
    "HardConstraints",
    "Candidate",
    "apply_mutations",
    "filter_by_distance",
    "upper_confidence_bound",
    "thompson_sample",
    "composite_objective",
    "rank_candidates",
    "compute_acquisition",
]
