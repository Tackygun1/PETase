from __future__ import annotations

from typing import Dict, List, Optional


from src.scoring.ddg_rosetta import load_ddg_table, score_sequence
from src.scoring.docking_md import activity_proxy, load_docking_scores, load_md_contacts

__all__ = [
    "hamming_distance",
    "mutation_list",
    "score_stability",
    "score_activity",
    "set_reference_sequence",
    "get_reference_sequence",
    "load_ddg_reference",
    "load_docking_reference",
]

_WT_SEQUENCE: str = ""
_DDG_TABLE = None
_DOCKING: Dict[str, float] = {}
_CONTACTS: Dict[str, float] = {}


def set_reference_sequence(seq: str) -> None:
    """Set the reference wild-type sequence used by scoring helpers."""
    global _WT_SEQUENCE
    _WT_SEQUENCE = seq or ""


def get_reference_sequence() -> str:
    return _WT_SEQUENCE


def load_ddg_reference(path) -> None:
    """Load a mutation->ΔΔG table for stability scoring."""
    global _DDG_TABLE
    _DDG_TABLE = load_ddg_table(path)


def load_docking_reference(docking_path=None, contacts_path=None) -> None:
    """Load docking and optional MD contacts tables for activity scoring."""
    global _DOCKING, _CONTACTS
    if docking_path:
        _DOCKING = load_docking_scores(docking_path)
    if contacts_path:
        _CONTACTS = load_md_contacts(contacts_path)


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
    Stability proxy using ΔΔG estimates; negative is stabilizing.
    """
    ref = _WT_SEQUENCE
    if not ref:
        raise ValueError("Reference sequence not set; call set_reference_sequence first.")
    return float(score_sequence(seq, ref, ddg_table=_DDG_TABLE))


def score_activity(seq: str) -> Optional[float]:
    """
    Activity proxy: docking/MD-derived if tables loaded, else heuristic based on aromatic/hydrophobic content.
    """
    if _DOCKING:
        # without sequence ID mapping, fall back to heuristic; sequence-level activity would need consistent IDs
        # here we approximate by using sequence string as key if present
        key = seq
        if key in _DOCKING:
            return float(activity_proxy(key, _DOCKING, _CONTACTS))
    # Heuristic: favor balanced hydrophobicity/aromatics near PET binding (F/Y/W, I/L/V)
    aromatic = sum(seq.count(x) for x in "FYW")
    hydrophobic = sum(seq.count(x) for x in "ILV")
    charge = sum(seq.count(x) for x in "KR") - sum(seq.count(x) for x in "DE")
    length = max(len(seq), 1)
    score = (aromatic + 0.5 * hydrophobic) / length - 0.01 * abs(charge)
    return float(score)
