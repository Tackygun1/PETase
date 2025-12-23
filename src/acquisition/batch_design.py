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
    ):
        self.embedding_path = embedding_path
        self.sequence_db = _load_sequence_db(sequence_db)
        self.wt_id = wt_id
        self.trust_radius = trust_radius
        self.embeddings = self._load_embeddings()
        if wt_id not in self.sequence_db:
            raise ValueError(f"Wild-type id {wt_id} not found in sequence DB.")
        self.wt_seq = self.sequence_db[wt_id]
        self.knn = self._fit_nn()

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        npz = np.load(self.embedding_path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    def _fit_nn(self) -> NearestNeighbors:
        ids, mats = zip(*[(k, v) for k, v in self.embeddings.items() if k in self.sequence_db])
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

    def propose(self, k: int = 32) -> List[MutationProposal]:
        neighbors = self.retrieve_neighbors(k=min(2 * k, len(self._emb_ids)))
        proposals: List[MutationProposal] = []
        rng = np.random.default_rng()

        for nb_id in neighbors:
            nb_seq = self.sequence_db[nb_id]
            muts = self._diff(self.wt_seq, nb_seq)
            if not muts:
                continue
            # sample subset within trust radius
            max_mut = rng.integers(self.trust_radius[0], self.trust_radius[1] + 1)
            chosen = muts[:max_mut]
            new_seq = list(self.wt_seq)
            for pos, _, alt in chosen:
                new_seq[pos] = alt
            proposals.append(
                MutationProposal(
                    seq_id=f"{self.wt_id}_x_{nb_id}",
                    sequence="".join(new_seq),
                    mutations=[(p + 1, ref, alt) for p, ref, alt in chosen],
                    source_neighbor=nb_id,
                    mutation_count=len(chosen),
                )
            )
            if len(proposals) >= k:
                break
        return proposals

    def _diff(self, a: str, b: str) -> List[Tuple[int, str, str]]:
        if len(a) != len(b):
            raise ValueError("Sequences must be same length for diffing.")
        muts = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
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
