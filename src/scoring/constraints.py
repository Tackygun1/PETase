"""
Structural and sequence constraints for PETase engineering.
"""

from __future__ import annotations

from typing import Iterable, Set

PROTECTED_POSITIONS: Set[int] = {
    160,
    206,
    237,  # catalytic triad
    87,
    161,
    185,  # oxyanion hole / cleft aromatics
    203,
    239,
    273,
    289,  # disulfides
}


def is_allowed_position(pos: int, extra_protected: Iterable[int] | None = None) -> bool:
    protected = set(PROTECTED_POSITIONS)
    if extra_protected:
        protected.update(extra_protected)
    return pos not in protected


def violates_motif(seq: str, motif: str = "NXS/T") -> bool:
    """Detect N-X-S/T glycosylation motif."""
    if len(seq) < 3:
        return False
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 2] in ("S", "T"):
            return True
    return False


__all__ = ["is_allowed_position", "violates_motif", "PROTECTED_POSITIONS"]
