"""Retrieval-augmented ESM prior for thermostability.

Given a reference bank of thermostable (or labeled) sequences with ESM embeddings, we score new
sequences by soft top-k similarity and return a prior stability score. This plugs into acquisition as the
`ram_score` term.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from ..utils.io import load_embeddings, load_labels_csv


def _normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norm


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x / max(temperature, 1e-6)) - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-8)


def _align_ref(emb_path: Path, labels_path: Path, id_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    ref_emb = load_embeddings(emb_path)
    ref_labels_df = load_labels_csv(labels_path, id_col=id_col, y_col=y_col)
    kept_ids: List[str] = [rid for rid in ref_labels_df[id_col].tolist() if rid in ref_emb]
    if not kept_ids:
        raise ValueError("No overlap between reference embeddings and labels.")
    mat = np.vstack([ref_emb[rid] for rid in kept_ids])
    labels = ref_labels_df.set_index(id_col).loc[kept_ids, y_col].to_numpy(dtype=float)
    return mat, labels


def build_ram_scorer(
    ref_embeddings_path: Path,
    ref_labels_path: Path,
    id_col: str = "id",
    y_col: str = "stability",
    top_k: int = 5,
    temperature: float = 0.1,
):
    """Return a scorer callable: score(candidates, cand_emb_lookup) -> np.ndarray."""
    ref_mat, ref_labels = _align_ref(ref_embeddings_path, ref_labels_path, id_col, y_col)
    ref_mat = _normalize(ref_mat)

    def score(candidates, cand_emb_lookup) -> np.ndarray:
        scores: List[float] = []
        for cand in candidates:
            emb = cand_emb_lookup.get(cand.seq_id)
            if emb is None:
                scores.append(0.0)
                continue
            ce = emb / (np.linalg.norm(emb) + 1e-8)
            sims = ref_mat @ ce  # cosine similarity since both normalized
            k = min(top_k, sims.shape[0])
            top_idx = np.argsort(sims)[::-1][:k]
            weights = _softmax(sims[top_idx], temperature=temperature)
            scores.append(float(np.sum(weights * ref_labels[top_idx])))
        return np.array(scores, dtype=float)

    return score
