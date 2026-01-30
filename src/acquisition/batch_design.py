"""
Batch design utilities: mutation proposal + stratified selection with QD archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .acquisition import HardConstraints, mutation_distance, rank_candidates
from ..scoring.constraints import is_allowed_position
from .qd_archive import CandidateRecord, QDArchive


def _load_sequence_db(seq_path: Path) -> Dict[str, str]:
    df = pd.read_csv(seq_path)
    if not {"id", "sequence"}.issubset(df.columns):
        raise ValueError("Sequence DB must contain columns: id, sequence")
    return dict(zip(df["id"], df["sequence"]))


@dataclass
class MutationProposal:
    seq_id: str
    sequence: str
    mutations: List[Tuple[int, str, str]]
    source_neighbor: str
    mutation_count: int


class MutationModel:
    """
    Retrieval-augmented mutation generator guided by ESM embeddings and trust regions.
    """

    def __init__(
        self,
        embedding_path: Path,
        sequence_db: Path,
        wt_id: str,
        trust_radius: Tuple[int, int] = (1, 5),
        extra_protected: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        self.embedding_path = embedding_path
        self.sequence_db = _load_sequence_db(sequence_db)
        self.wt_id = wt_id
        self.trust_radius = trust_radius
        self.extra_protected = list(extra_protected) if extra_protected else None
        self._rng = np.random.default_rng(seed)
        self.embeddings = self._load_embeddings()
        if wt_id not in self.sequence_db:
            raise ValueError(f"Wild-type id {wt_id} not found in sequence DB.")
        self.wt_seq = self.sequence_db[wt_id]
        self.knn = self._fit_nn()

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        npz = np.load(self.embedding_path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    def _fit_nn(self) -> NearestNeighbors:
        pairs = [
            (k, v)
            for k, v in self.embeddings.items()
            if k in self.sequence_db and len(self.sequence_db[k]) == len(self.wt_seq)
        ]
        if not pairs:
            raise ValueError("No embeddings found with the same length as the WT sequence.")
        ids, mats = zip(*pairs)
        self._emb_ids = list(ids)
        X = np.stack(mats)
        nn = NearestNeighbors(metric="cosine")
        nn.fit(X)
        return nn

    def retrieve_neighbors(self, k: int = 8) -> List[str]:
        query = self.embeddings[self.wt_id]
        dists, idx = self.knn.kneighbors(query[None, :], n_neighbors=min(k, len(self._emb_ids)))
        neighbors = [self._emb_ids[i] for i in idx[0] if self._emb_ids[i] != self.wt_id]
        return neighbors

    def propose(
        self,
        k: int = 32,
        proposals_per_neighbor: int = 1,
        proposal_mode: str = "neighbor",
        seed: int | None = None,
    ) -> List[MutationProposal]:
        neighbors = self.retrieve_neighbors(k=min(5 * k, len(self._emb_ids)))
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        rng = rng or np.random.default_rng()
        rng.shuffle(neighbors)

        if proposal_mode == "pool":
            proposals = self._propose_from_pool(neighbors, k=k, rng=rng)
            if len(proposals) >= k:
                return proposals
            # fall back to neighbor subsets if pool is small
            remaining = k - len(proposals)
            proposals.extend(
                self._propose_from_neighbors(
                    neighbors, k=remaining, rng=rng, proposals_per_neighbor=proposals_per_neighbor
                )
            )
            return proposals

        return self._propose_from_neighbors(
            neighbors, k=k, rng=rng, proposals_per_neighbor=proposals_per_neighbor
        )

    def _propose_from_neighbors(
        self,
        neighbors: List[str],
        k: int,
        rng: np.random.Generator,
        proposals_per_neighbor: int,
    ) -> List[MutationProposal]:
        proposals: List[MutationProposal] = []
        seen = set()
        for nb_id in neighbors:
            nb_seq = self.sequence_db[nb_id]
            muts = self._diff(self.wt_seq, nb_seq)
            if not muts:
                continue
            min_mut, max_mut = self.trust_radius
            if len(muts) < min_mut:
                continue
            max_mut = min(max_mut, len(muts))
            for _ in range(max(1, proposals_per_neighbor)):
                size = int(rng.integers(min_mut, max_mut + 1))
                choice_idx = rng.choice(len(muts), size=size, replace=False)
                chosen = [muts[i] for i in sorted(choice_idx)]
                new_seq = list(self.wt_seq)
                for pos, _, alt in chosen:
                    new_seq[pos] = alt
                seq = "".join(new_seq)
                if seq in seen:
                    continue
                seen.add(seq)
                proposals.append(
                    MutationProposal(
                        seq_id=f"{self.wt_id}_x_{nb_id}_{len(proposals)}",
                        sequence=seq,
                        mutations=[(p + 1, ref, alt) for p, ref, alt in chosen],
                        source_neighbor=nb_id,
                        mutation_count=len(chosen),
                    )
                )
                if len(proposals) >= k:
                    break
            if len(proposals) >= k:
                break
        return proposals

    def _propose_from_pool(
        self, neighbors: List[str], k: int, rng: np.random.Generator
    ) -> List[MutationProposal]:
        pool = self._build_diff_pool(neighbors)
        positions = list(pool.keys())
        if not positions:
            return []
        min_mut, max_mut = self.trust_radius
        max_mut = min(max_mut, len(positions))
        proposals: List[MutationProposal] = []
        seen = set()
        attempts = 0
        max_attempts = max(k * 20, 200)

        while len(proposals) < k and attempts < max_attempts:
            attempts += 1
            size = int(rng.integers(min_mut, max_mut + 1))
            chosen_pos = rng.choice(positions, size=size, replace=False)
            new_seq = list(self.wt_seq)
            muts: List[Tuple[int, str, str]] = []
            for pos in chosen_pos:
                alt = rng.choice(list(pool[pos]))
                ref = self.wt_seq[pos]
                new_seq[pos] = alt
                muts.append((pos + 1, ref, alt))
            seq = "".join(new_seq)
            if seq in seen or seq == self.wt_seq:
                continue
            seen.add(seq)
            proposals.append(
                MutationProposal(
                    seq_id=f"{self.wt_id}_pool_{len(proposals)}",
                    sequence=seq,
                    mutations=muts,
                    source_neighbor="pool",
                    mutation_count=len(muts),
                )
            )
        return proposals

    def _build_diff_pool(self, neighbors: List[str]) -> Dict[int, List[str]]:
        pool: Dict[int, set[str]] = {}
        for nb_id in neighbors:
            nb_seq = self.sequence_db[nb_id]
            if len(nb_seq) != len(self.wt_seq):
                continue
            for i, (x, y) in enumerate(zip(self.wt_seq, nb_seq)):
                if x == y:
                    continue
                if not is_allowed_position(i + 1, self.extra_protected):
                    continue
                pool.setdefault(i, set()).add(y)
        return {k: sorted(list(v)) for k, v in pool.items()}

    def _diff(self, a: str, b: str) -> List[Tuple[int, str, str]]:
        if len(a) != len(b):
            return []
        muts = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                if is_allowed_position(i + 1, self.extra_protected):
                    muts.append((i, x, y))
        return muts


@dataclass
class BatchDesignConfig:
    batch_size: int = 12
    min_hamming: int = 2
    beta_start: float = 2.0
    beta_end: float = 0.5


class BatchDesigner:
    """
    Selects a batch using acquisition ranking + QD stratification + diversity control.
    """

    def __init__(
        self,
        cfg: BatchDesignConfig,
        constraints: HardConstraints,
        qd_archive: QDArchive,
        wt_seq: str,
    ):
        self.cfg = cfg
        self.constraints = constraints
        self.qd = qd_archive
        self.wt_seq = wt_seq

    def select(
        self,
        sequences: Sequence[str],
        seq_ids: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
        stability_scores: np.ndarray,
        activity_scores: np.ndarray,
        round_idx: int = 0,
        meta_lookup: Dict[str, Dict] | None = None,
    ) -> List[CandidateRecord]:
        beta = np.linspace(self.cfg.beta_start, self.cfg.beta_end, num=max(round_idx + 2, 2))[
            round_idx
        ]
        order = rank_candidates(
            mean=mean,
            std=std,
            wt_seq=self.wt_seq,
            sequences=sequences,
            constraints=self.constraints,
            beta=beta,
        )

        chosen: List[int] = []
        for idx in order:
            if len(chosen) >= self.cfg.batch_size:
                break
            seq = sequences[idx]
            if any(mutation_distance(seq, sequences[j]) < self.cfg.min_hamming for j in chosen):
                continue
            chosen.append(idx)

        records: List[CandidateRecord] = []
        for idx in chosen:
            mut_count = mutation_distance(sequences[idx], self.wt_seq)
            comp = stability_scores[idx] + 0.5 * activity_scores[idx]
            meta = {"acquisition_order": order.index(idx), "round": round_idx}
            if meta_lookup and seq_ids[idx] in meta_lookup:
                meta.update(meta_lookup[seq_ids[idx]])
            records.append(
                CandidateRecord(
                    seq_id=seq_ids[idx],
                    sequence=sequences[idx],
                    mutation_count=mut_count,
                    stability_score=float(stability_scores[idx]),
                    activity_score=float(activity_scores[idx]),
                    composite=float(comp),
                    meta=meta,
                )
            )
        # Insert into QD archive and return elites
        self.qd.batch_insert(records)
        return records


__all__ = ["MutationModel", "MutationProposal", "BatchDesigner", "BatchDesignConfig"]
