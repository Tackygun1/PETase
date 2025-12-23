"""
Lightweight Quality-Diversity archive for PETase variant design.

Bins candidates along (mutation_count, stability_bin) axes and stores per-niche elites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class CandidateRecord:
    seq_id: str
    sequence: str
    mutation_count: int
    stability_score: float
    activity_score: float
    composite: float
    meta: Dict


class QDArchive:
    def __init__(
        self,
        max_mutations: int = 10,
        stability_bins: Iterable[float] | None = None,
        stability_bin_width: float | None = None,
        top_k_per_niche: int = 1,
    ):
        self.max_mutations = max_mutations
        if stability_bins is None and stability_bin_width is not None:
            # symmetric bins around 0 up to +/- max_mutations for simplicity
            span = max_mutations + 2
            stability_bins = np.arange(-span, span + stability_bin_width, stability_bin_width)
        self.stability_bins = np.array(
            stability_bins if stability_bins is not None else [-2, -1, 0, 1, 2, 4]
        )
        self.top_k = top_k_per_niche
        # mapping: (mutation_bin, stability_bin) -> list[CandidateRecord]
        self._archive: Dict[Tuple[int, int], List[CandidateRecord]] = {}

    def _niche(self, mutation_count: int, stability: float) -> Tuple[int, int]:
        mut_bin = min(mutation_count, self.max_mutations)
        stab_bin = int(np.digitize([stability], self.stability_bins, right=False)[0])
        return mut_bin, stab_bin

    def insert(self, cand: CandidateRecord):
        key = self._niche(cand.mutation_count, cand.stability_score)
        bucket = self._archive.setdefault(key, [])
        bucket.append(cand)
        bucket.sort(key=lambda c: c.composite, reverse=True)
        self._archive[key] = bucket[: self.top_k]

    def maybe_insert(self, mut_count: int, stability: float, acquisition: float, meta: Dict):
        """Compatibility helper for older tests: mut_count/stability/acquisition -> store as CandidateRecord."""
        rec = CandidateRecord(
            seq_id=meta.get("seq_id", f"cand_{len(self._archive)}"),
            sequence=meta.get("sequence", ""),
            mutation_count=mut_count,
            stability_score=stability,
            activity_score=meta.get("activity_score", 0.0),
            composite=acquisition,
            meta=meta,
        )
        self.insert(rec)

    def batch_insert(self, cands: Iterable[CandidateRecord]):
        for c in cands:
            self.insert(c)

    def elites(self) -> List[CandidateRecord]:
        out: List[CandidateRecord] = []
        for bucket in self._archive.values():
            out.extend(bucket)
        return out

    def stratified_sample(self, k: int) -> List[CandidateRecord]:
        elites = self.elites()
        if not elites:
            return []
        # Sample across niches in round-robin fashion
        niches = list(self._archive.keys())
        picks: List[CandidateRecord] = []
        idx = 0
        while len(picks) < k and idx < len(elites) * 2:
            niche = niches[idx % len(niches)]
            bucket = self._archive[niche]
            picks.append(bucket[idx % len(bucket)])
            idx += 1
        return picks[:k]

    def coverage(self) -> float:
        filled = len(self._archive)
        total = (self.max_mutations + 1) * (len(self.stability_bins) + 1)
        return filled / total


__all__ = ["CandidateRecord", "QDArchive"]
