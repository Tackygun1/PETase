"""Acquisition scoring for mutation picking with BO + QD.

Supports two modes:
1) Surrogate-only UCB (baseline).
2) Surrogate UCB blended with a RAM-ESM prior for thermostability-aware selection.

This module does not run heavy models itself; it expects callers to provide predictions or lightweight
callables that fit inside local constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import numpy as np


# -------------------------
# Data structures
# -------------------------
@dataclass
class Candidate:
    seq_id: str
    sequence: str
    mutations: List[str]  # e.g., ["A87Y", "R280A"]
    mut_count: int
    pred_stab_mean: float
    pred_stab_std: float
    pred_act_mean: Optional[float] = None
    ram_score: Optional[float] = None
    acquisition: Optional[float] = None
    meta: dict = field(default_factory=dict)


# -------------------------
# Scoring
# -------------------------
def ucb(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    return mean + beta * std


def compute_acquisition(
    candidates: Iterable[Candidate],
    beta: float = 1.0,
    w_stability: float = 1.0,
    w_activity: float = 0.0,
    w_ram: float = 0.0,
    activity_floor: Optional[float] = None,
) -> List[Candidate]:
    """Assign acquisition scores in-place and return the list.

    w_ram=0 yields the surrogate-only version; w_ram>0 blends in RAM-ESM prior scores.
    """
    updated: List[Candidate] = []
    for c in candidates:
        stab = w_stability * (ucb(np.array([c.pred_stab_mean]), np.array([c.pred_stab_std]), beta)[0])

        act_term = 0.0
        if c.pred_act_mean is not None and w_activity != 0:
            act_term = w_activity * c.pred_act_mean
            if activity_floor is not None and c.pred_act_mean < activity_floor:
                # Hard filter below the activity floor.
                c.acquisition = -np.inf
                updated.append(c)
                continue

        ram_term = 0.0
        if c.ram_score is not None and w_ram != 0:
            ram_term = w_ram * c.ram_score

        c.acquisition = stab + act_term + ram_term
        updated.append(c)
    return updated


# -------------------------
# Candidate utilities
# -------------------------
def apply_mutations(parent_seq: str, mutations: List[str]) -> str:
    """Apply mutation strings like 'A87Y' onto parent_seq (1-based index)."""
    seq = list(parent_seq)
    for mut in mutations:
        if len(mut) < 3:
            raise ValueError(f"Bad mutation format: {mut}")
        orig = mut[0]
        dest = mut[-1]
        try:
            pos = int(mut[1:-1])
        except ValueError as e:
            raise ValueError(f"Bad mutation position in {mut}") from e
        if pos < 1 or pos > len(seq):
            raise ValueError(f"Position out of range in {mut} for sequence length {len(seq)}")
        if seq[pos - 1] != orig:
            raise ValueError(f"Origin mismatch at {mut}: seq has {seq[pos-1]}")
        seq[pos - 1] = dest
    return "".join(seq)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length for Hamming distance.")
    return sum(x != y for x, y in zip(a, b))


def filter_by_distance(candidates: List[Candidate], min_hamming: int) -> List[Candidate]:
    """Greedy distance filter to keep diverse picks."""
    kept: List[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.acquisition or -np.inf, reverse=True):
        if all(hamming(cand.sequence, k.sequence) >= min_hamming for k in kept):
            kept.append(cand)
    return kept


# -------------------------
# RAM-ESM adapter (optional)
# -------------------------
def ram_esm_wrapper(ram_model: Callable[[List[str]], np.ndarray], sequences: List[str]) -> List[float]:
    """Run an external RAM-ESM scorer on sequences. Expects a callable returning a numpy array."""
    scores = ram_model(sequences)
    return [float(s) for s in scores]
