"""Hard mutation constraints derived from PETase structural knowledge.

These are conservative: catalytic triad, oxyanion hole, aromatic clamp, and disulfide cysteines are
protected from mutation by default.
"""

from __future__ import annotations

from typing import Set

# 1-based residue indices on the FAST-PETase/PETase numbering.
CATALYTIC_TRIAD: Set[int] = {160, 206, 237}  # Ser160, Asp206, His237
OXYANION_HOLE: Set[int] = {87, 161}  # Tyr87, Met161 backbone NH
AROMATIC_CLAMP: Set[int] = {87, 159, 185, 241}  # Tyr87, Trp159, Trp185, Asn241 neighbors
DISULFIDES: Set[int] = {203, 239, 273, 289}  # Cys203-Cys239, Cys273-Cys289

PROTECTED_SITES: Set[int] = CATALYTIC_TRIAD | OXYANION_HOLE | AROMATIC_CLAMP | DISULFIDES


def is_allowed_position(position: int) -> bool:
    """Return True if the position can be mutated under hard constraints."""
    return position not in PROTECTED_SITES
