from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = [
    "hamming_distance",
    "mutation_list",
    "score_stability",
    "score_activity",
    "set_reference_sequence",
    "get_reference_sequence",
]

_WT_SEQUENCE: str = ""


def set_reference_sequence(seq: str) -> None:
    """Set the reference wild-type sequence used by scoring helpers."""
    global _WT_SEQUENCE
    _WT_SEQUENCE = seq or ""


def get_reference_sequence() -> str:
    return _WT_SEQUENCE


def hamming_distance(a: str, b: str) -> int:
    if not a or not b:
        raise ValueError("Sequences must be non-empty for Hamming distance.")
    if len(a) != len(b):
        raise ValueError("Sequences must have the same length for Hamming distance.")
    return sum(x != y for x, y in zip(a, b))


def mutation_list(parent: str, child: str) -> List[str]:
    """Return mutations as e.g. ['S121E', 'D186H']."""
    muts = []
    for i, (aa0, aa1) in enumerate(zip(parent, child), start=1):
        if aa0 != aa1:
            muts.append(f"{aa0}{i}{aa1}")
    return muts


def score_stability(seq: str) -> float:
    """
    Placeholder for a proper stability predictor. Higher is better.
    Currently: penalize mutation count plus Gaussian noise.
    """
    ref = _WT_SEQUENCE
    mut_count = hamming_distance(seq, ref) if ref else 0
    return -mut_count + np.random.normal(scale=0.2)


def score_activity(seq: str) -> Optional[float]:
    """
    Placeholder. Replace with docking/ML predictor once available.
    Return None if activity is not factored yet.
    """
    return None
