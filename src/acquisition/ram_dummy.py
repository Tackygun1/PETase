"""Lightweight RAM-ESM stub.

Exports score(List[str]) -> np.ndarray for integration testing without heavy models.
Scores favor longer sequences slightly to break ties, but remain cheap and deterministic.
"""

from __future__ import annotations

from typing import List

import numpy as np


def score(sequences: List[str]) -> np.ndarray:
    # Simple heuristic: length + normalized amino acid diversity.
    scores = []
    for seq in sequences:
        length_term = len(seq) * 1e-3
        diversity_term = len(set(seq)) / 30.0
        scores.append(length_term + diversity_term)
    return np.array(scores, dtype=float)
