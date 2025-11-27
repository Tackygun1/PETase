"""Simple quality-diversity archive for PETase variant search.

We discretize by mutation count (x-axis) and predicted stability bin (y-axis) and keep one elite per
niche. This is intentionally lightweight so it can back both RAM-ESM and non-RAM acquisition flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ArchiveEntry:
    niche: Tuple[int, int]
    score: float
    payload: dict


class QDArchive:
    def __init__(self, max_mutations: int, stability_bin_width: float = 0.5):
        """
        Parameters
        ----------
        max_mutations : int
            Maximum mutation count to index niches on the x-axis.
        stability_bin_width : float
            Width of stability bins (y-axis). For example, width=0.5 groups predictions as [..., -0.5, 0, 0.5...].
        """
        self.max_mutations = max_mutations
        self.stability_bin_width = stability_bin_width
        self._grid: Dict[Tuple[int, int], ArchiveEntry] = {}

    def _niche(self, mutation_count: int, stability_score: float) -> Tuple[int, int]:
        m_bin = min(mutation_count, self.max_mutations)
        s_bin = int(stability_score / self.stability_bin_width)
        return m_bin, s_bin

    def maybe_insert(self, mutation_count: int, stability_score: float, score: float, payload: dict) -> None:
        """Insert into archive if empty or better than existing entry for the niche."""
        key = self._niche(mutation_count, stability_score)
        existing = self._grid.get(key)
        if existing is None or score > existing.score:
            self._grid[key] = ArchiveEntry(niche=key, score=score, payload=payload)

    def elites(self):
        """Return archive entries sorted by score descending."""
        return sorted(self._grid.values(), key=lambda e: e.score, reverse=True)

    def snapshot(self) -> Dict[str, dict]:
        """Export a JSON-serializable snapshot."""
        return {
            f"{k[0]}_{k[1]}": {"score": v.score, **v.payload}
            for k, v in self._grid.items()
        }
