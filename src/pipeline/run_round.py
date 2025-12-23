"""
End-to-end round runner:
- Propose mutations with retrieval-guided MutationModel.
- Score candidates with the surrogate ensemble (stability/activity).
- Select a diverse batch via BatchDesigner + QD archive under hard constraints.

Requires:
    - Sequence DB CSV with columns: id, sequence (must include the wild-type ID).
    - Embedding NPZ keyed by sequence IDs (used for retrieval; can also be reused as surrogate features).
    - Trained surrogate .pt from src.models.surrogate.train_from_config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..acquisition.batch_design import BatchDesignConfig, BatchDesigner, MutationModel
from ..acquisition.acquisition import HardConstraints
from ..acquisition.qd_archive import QDArchive
from ..models.surrogate import SurrogateEnsemble
from ..petase_models import one_hot_encode  # fallback featurizer


# --------------------------
# Helpers
# --------------------------
def read_fasta_single(path: Path) -> Tuple[str, str]:
    header = None
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip()
            else:
                seq.append(line)
    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    return header, "".join(seq)


def build_constraints(
    wt_seq: str, max_mutations: int, forbid_motif: Sequence[str]
) -> HardConstraints:
    # Map key catalytic/binding residues to WT amino acids (1-based indexing)
    positions = [160, 206, 237, 87, 161, 185]
    catalytic_positions: Dict[int, str] = {}
    for pos in positions:
        if pos > len(wt_seq):
            continue
        catalytic_positions[pos] = wt_seq[pos - 1]
    disulfide_pairs = [(203, 239), (273, 289)]
    return HardConstraints(
        catalytic_positions=catalytic_positions,
        disulfide_pairs=disulfide_pairs,
        forbid_motif=forbid_motif,
        max_mutations=max_mutations,
    )


def embed_sequences(
    seq_ids: Sequence[str],
    sequences: Sequence[str],
    embeddings: Dict[str, np.ndarray],
    embedder: str = "precomputed",
) -> np.ndarray:
    """
    Return a feature matrix aligned to seq_ids. If an id is missing:
      - with embedder='onehot', compute flattened one-hot (len*20) vectors
      - with embedder='precomputed', raise
    """
    feats: List[np.ndarray] = []
    for sid, seq in zip(seq_ids, sequences):
        emb = embeddings.get(sid)
        if emb is None:
            if embedder != "onehot":
                raise ValueError(f"Missing embedding for {sid}; provide an embedder or precompute.")
            emb_vec = one_hot_encode(seq)
            feats.append(emb_vec)
        else:
            feats.append(emb)

    # ensure consistent dims
    dims = {f.shape[0] for f in feats}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions detected: {dims}")
    return np.vstack(feats)


def load_embeddings_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# --------------------------
# Round orchestration
# --------------------------
def run_round(
    parent_fasta: Path,
    sequence_db: Path,
    embeddings_npz: Path,
    surrogate_path: Path,
    batch_size: int = 12,
    proposals: int = 64,
    trust_min: int = 1,
    trust_max: int = 5,
    forbid_motif: Sequence[str] = ("N", "X", "S", "T"),  # avoid N-X-S/T motifs
    embedder: str = "onehot",
    stability_bins: Sequence[float] = (-2, -1, 0, 1, 2, 4),
) -> List:
    wt_id, wt_seq = read_fasta_single(parent_fasta)
    embeddings = load_embeddings_npz(embeddings_npz)

    mut_model = MutationModel(
        embedding_path=embeddings_npz,
        sequence_db=sequence_db,
        wt_id=wt_id,
        trust_radius=(trust_min, trust_max),
    )
    proposals = mut_model.propose(k=proposals)

    seq_ids = [p.seq_id for p in proposals]
    seqs = [p.sequence for p in proposals]
    meta_lookup = {
        p.seq_id: {"mutations": p.mutations, "source_neighbor": p.source_neighbor}
        for p in proposals
    }

    features = embed_sequences(seq_ids, seqs, embeddings, embedder=embedder)

    surrogate = SurrogateEnsemble.load(surrogate_path)
    mean, var = surrogate.predict(features)
    std = np.sqrt(np.clip(var, 1e-8, None))

    constraints = build_constraints(wt_seq, max_mutations=trust_max, forbid_motif=forbid_motif)
    archive = QDArchive(max_mutations=trust_max, stability_bins=stability_bins, top_k_per_niche=1)
    designer = BatchDesigner(
        cfg=BatchDesignConfig(batch_size=batch_size, min_hamming=2, beta_start=2.0, beta_end=0.5),
        constraints=constraints,
        qd_archive=archive,
        wt_seq=wt_seq,
    )

    stability_scores = mean[:, 0]
    activity_scores = mean[:, 1] if mean.shape[1] > 1 else np.zeros_like(stability_scores)

    selected = designer.select(
        sequences=seqs,
        seq_ids=seq_ids,
        mean=mean,
        std=std,
        stability_scores=stability_scores,
        activity_scores=activity_scores,
        round_idx=0,
        meta_lookup=meta_lookup,
    )
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run mutation proposal + surrogate + QD selection.")
    p.add_argument(
        "--parent-fasta",
        type=Path,
        required=True,
        help="FASTA with the wild-type sequence (first record used).",
    )
    p.add_argument(
        "--sequence-db",
        type=Path,
        required=True,
        help="CSV with columns id,sequence (must include the WT id).",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="NPZ with embeddings keyed by id (used for retrieval/features).",
    )
    p.add_argument(
        "--surrogate", type=Path, required=True, help="Path to trained surrogate .pt file."
    )
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--proposals", type=int, default=64)
    p.add_argument(
        "--trust-min",
        type=int,
        default=1,
        help="Minimum mutations from retrieved neighbor (inclusive).",
    )
    p.add_argument(
        "--trust-max",
        type=int,
        default=5,
        help="Maximum mutations from retrieved neighbor (inclusive).",
    )
    p.add_argument(
        "--embedder",
        type=str,
        default="onehot",
        choices=["onehot", "precomputed"],
        help="How to featurize sequences for the surrogate if missing in NPZ.",
    )
    p.add_argument(
        "--stability-bins",
        type=str,
        default="-2,-1,0,1,2,4",
        help="Comma-separated stability bin edges for QD archive.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bins = tuple(float(x.strip()) for x in args.stability_bins.split(",") if x.strip())
    selected = run_round(
        parent_fasta=args.parent_fasta,
        sequence_db=args.sequence_db,
        embeddings_npz=args.embeddings,
        surrogate_path=args.surrogate,
        batch_size=args.batch_size,
        proposals=args.proposals,
        trust_min=args.trust_min,
        trust_max=args.trust_max,
        embedder=args.embedder,
        stability_bins=bins,
    )
    for rec in selected:
        print(
            f"{rec.seq_id}\tmut_count={rec.mutation_count}\tstab_pred={rec.stability_score:.3f}\tactivity_pred={rec.activity_score:.3f}"
        )


if __name__ == "__main__":
    main()
