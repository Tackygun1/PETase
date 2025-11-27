from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize

__all__ = [
    "AMINO_ACIDS",
    "one_hot_encode",
    "make_one_hot_encoder",
    "load_esm_embedder",
    "SurrogateModel",
    "EmbeddingCache",
    "build_known_petase_index",
    "KNOWN_PETASE_INDEX",
    "KNOWN_PETASE_EMBEDDINGS",
]

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_ESM_EMBEDDER_CACHE: Dict[Tuple[str, str, int], Callable[[List[str]], np.ndarray]] = {}


def one_hot_encode(seq: str) -> np.ndarray:
    """Simple one-hot per position; returns (L, 20) flattened to (L*20,)."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    L = len(seq)
    mat = np.zeros((L, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            mat[i, aa_to_idx[aa]] = 1.0
    return mat.reshape(-1)


def make_one_hot_encoder() -> Callable[[List[str]], np.ndarray]:
    """Return a callable that encodes a list of sequences to one-hot vectors."""

    def _encode(seqs: List[str]) -> np.ndarray:
        return np.vstack([one_hot_encode(s) for s in seqs])

    return _encode


def load_esm_embedder(
    model_name: str = "esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    batch_size: int = 8,
):
    """
    Lazily load an ESM model and return a callable that embeds sequences to vectors.

    Requires:
        pip install fair-esm torch
    """
    import torch  # noqa: F401 - optional dependency
    import esm

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, resolved_device, batch_size)
    if key in _ESM_EMBEDDER_CACHE:
        return _ESM_EMBEDDER_CACHE[key]

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval().to(resolved_device)
    batch_converter = alphabet.get_batch_converter()
    layer = model.num_layers

    def _encode(seqs: List[str]) -> np.ndarray:
        data = [(str(i), s) for i, s in enumerate(seqs)]
        outputs = []
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            _, batch_strs, tokens = batch_converter(batch)
            tokens = tokens.to(resolved_device)
            with torch.no_grad():
                res = model(tokens, repr_layers=[layer], return_contacts=False)
            reps = res["representations"][layer]
            for i, seq in enumerate(batch_strs):
                seq_len = len(seq)
                emb = reps[i, 1 : seq_len + 1].mean(0).cpu().numpy()
                outputs.append(emb)
        return np.stack(outputs, axis=0)

    _ESM_EMBEDDER_CACHE[key] = _encode
    return _encode


class SurrogateModel:
    def __init__(self, encoder: Optional[Callable[[List[str]], np.ndarray]] = None):
        # Random forest gives some notion of uncertainty via per-tree variance
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
        self._is_fitted = False
        self._encode = encoder or make_one_hot_encoder()

    def fit(self, seqs: List[str], scores: List[float]):
        X = self._encode(seqs)
        y = np.array(scores, dtype=float)
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_with_uncertainty(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_fitted, "Surrogate not fitted yet"
        X = self._encode(seqs)
        # Use per-tree predictions to estimate variance
        all_tree_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=1)
        mean = all_tree_preds.mean(axis=1)
        std = all_tree_preds.std(axis=1)
        return mean, std


KNOWN_PETASE_INDEX: Dict[str, str] = {}
KNOWN_PETASE_EMBEDDINGS: Optional[np.ndarray] = None


class EmbeddingCache:
    """Disk-backed cache for sequence embeddings."""

    def __init__(
        self,
        embedder: Callable[[List[str]], np.ndarray],
        cache_path: Optional[str] = None,
    ):
        from pathlib import Path

        self.embedder = embedder
        self.path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, np.ndarray] = {}
        self._dirty = False
        if self.path and self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            seqs = data["seqs"]
            embs = data["embeddings"]
            self.cache = {str(seq): emb for seq, emb in zip(seqs.tolist(), embs)}

    def get(self, seqs: List[str]) -> np.ndarray:
        if not seqs:
            return np.zeros((0, 0), dtype=np.float32)
        missing = [s for s in seqs if s not in self.cache]
        if missing:
            new_embs = self.embedder(missing)
            for seq, emb in zip(missing, new_embs):
                self.cache[seq] = emb
            if self.path:
                self._dirty = True
        return np.stack([self.cache[s] for s in seqs], axis=0)

    def save(self):
        if not self.path or not self._dirty or not self.cache:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        seqs = np.array(list(self.cache.keys()))
        embs = np.stack(list(self.cache.values()), axis=0)
        np.savez_compressed(self.path, seqs=seqs, embeddings=embs)
        self._dirty = False


def build_known_petase_index(
    wt_seq: str,
    model_name: str = "esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    batch_size: int = 8,
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
):
    """
    Build an ESM embedding index from the given PETase WT sequence.
    You can later extend this with more sequences (e.g. FAST-PETase, homologs).
    """
    global KNOWN_PETASE_INDEX, KNOWN_PETASE_EMBEDDINGS

    embedder = embedder or load_esm_embedder(
        model_name=model_name, device=device, batch_size=batch_size
    )
    emb = embedder([wt_seq])  # shape (1, D)

    KNOWN_PETASE_INDEX = {"petase_wt": wt_seq}
    KNOWN_PETASE_EMBEDDINGS = normalize(emb)  # L2-normalized latent vector

    print("Built ESM index for known PETase variant(s) (currently: wild-type only)")
