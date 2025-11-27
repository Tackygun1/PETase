"""Embedding retrieval utilities for RAG-ESM style workflows."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_ref_embeddings(emb_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load reference embeddings .npz -> (matrix, id_list)."""
    data = np.load(emb_path, allow_pickle=False)
    ids = list(data.files)
    mat = np.vstack([data[i] for i in ids])
    return mat, ids


def normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norm


def cosine_search(
    query: np.ndarray,
    ref_mat: np.ndarray,
    ref_ids: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return (id, similarity) for top_k neighbors via cosine similarity."""
    q = query / (np.linalg.norm(query) + 1e-8)
    sims = ref_mat @ q
    idx = np.argsort(sims)[::-1][:top_k]
    return [(ref_ids[i], float(sims[i])) for i in idx]


def radius_filter(neighbors: List[Tuple[str, float]], min_sim: float) -> List[Tuple[str, float]]:
    """Keep neighbors above a similarity threshold."""
    return [(nid, sim) for nid, sim in neighbors if sim >= min_sim]
