"""End-to-end round runner: propose mutations, score with surrogate (+optional RAM-ESM),
and select a diverse batch via QD.

This stays light: it assumes embeddings are precomputed and stored in a .npz keyed by candidate IDs.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..acquisition import Candidate, design_batch
from ..acquisition.proposer import propose_mutations, propose_from_neighbors
from ..acquisition.acquisition import ram_esm_wrapper
from ..acquisition.ram_esm import build_ram_scorer
from ..acquisition.retrieval import cosine_search, load_ref_embeddings, normalize, radius_filter
from ..acquisition.qd_archive import QDArchive
from ..models.surrogate import SurrogateModel
from ..utils.io import load_embeddings


def _load_ram_callable(module_path: str) -> Callable[[List[str]], np.ndarray]:
    mod = importlib.import_module(module_path)
    scorer = getattr(mod, "score", None)
    if scorer is None:
        raise ValueError(f"RAM module {module_path} must expose a `score(List[str]) -> np.ndarray`")
    return scorer


def attach_surrogate_preds(model: SurrogateModel, embeddings: dict, candidates: List[Candidate]) -> None:
    """Mutates candidates in-place with surrogate mean/std. Missing embeddings are skipped."""
    X: List[np.ndarray] = []
    idx: List[int] = []
    for i, c in enumerate(candidates):
        emb = embeddings.get(c.seq_id)
        if emb is None:
            continue
        X.append(emb)
        idx.append(i)
    if not X:
        return
    X_mat = np.vstack(X)
    preds = model.predict(X_mat)
    for j, i in enumerate(idx):
        candidates[i].pred_stab_mean = float(preds[j])
        candidates[i].pred_stab_std = 0.0  # placeholder; replace with ensemble SD if available


def attach_ram_scores_from_embeddings(
    ram_scorer: Callable, candidates: List[Candidate], emb_lookup: dict
) -> None:
    scores = ram_scorer(candidates, emb_lookup)
    for c, s in zip(candidates, scores):
        c.ram_score = float(s)


def _load_ref_sequences(seq_path: Path) -> dict:
    """Load reference sequences from CSV/TSV with columns id,sequence."""
    df = pd.read_csv(seq_path)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Reference sequence file must have columns: id, sequence")
    return dict(zip(df["id"].astype(str), df["sequence"].astype(str)))


def run_round(
    parent_id: str,
    parent_seq: str,
    candidate_sites: Sequence[int],
    embeddings_path: Path,
    surrogate_dir: Path,
    batch_size: int = 8,
    max_mutations: int = 1,
    use_ram: bool = False,
    ram_module: Optional[str] = None,
    ram_ref_embeddings: Optional[Path] = None,
    ram_ref_labels: Optional[Path] = None,
    ram_top_k: int = 5,
    ram_temperature: float = 0.1,
    use_rag_neighbors: bool = False,
    rag_ref_embeddings: Optional[Path] = None,
    rag_ref_sequences: Optional[Path] = None,
    rag_top_k: int = 5,
    rag_min_similarity: float = 0.5,
) -> List[Candidate]:
    candidates = propose_mutations(
        parent_id=parent_id,
        parent_seq=parent_seq,
        candidate_sites=candidate_sites,
        max_mutations=max_mutations,
    )

    # RAG neighbor-derived proposals
    if use_rag_neighbors:
        if rag_ref_embeddings is None or rag_ref_sequences is None:
            raise ValueError("use_rag_neighbors requires rag_ref_embeddings and rag_ref_sequences")
        ref_mat, ref_ids = load_ref_embeddings(rag_ref_embeddings)
        ref_mat = normalize(ref_mat)
        ref_seq_map = _load_ref_sequences(rag_ref_sequences)
        if parent_id not in ref_seq_map or parent_id not in ref_ids:
            raise ValueError("Parent ID must exist in reference sequences/embeddings for RAG retrieval.")
        # Use parent embedding from reference bank
        parent_idx = ref_ids.index(parent_id)
        parent_emb = ref_mat[parent_idx]
        neighbors = cosine_search(parent_emb, ref_mat, ref_ids, top_k=rag_top_k + 1)  # +1 to include parent
        neighbors = [(nid, sim) for nid, sim in neighbors if nid != parent_id]
        neighbors = radius_filter(neighbors, min_sim=rag_min_similarity)
        neighbor_ids = [nid for nid, _ in neighbors]
        neighbor_seqs = [ref_seq_map[nid] for nid in neighbor_ids if nid in ref_seq_map]
        candidates += propose_from_neighbors(
            parent_id=parent_id,
            parent_seq=parent_seq,
            neighbors=neighbor_seqs,
            neighbor_ids=neighbor_ids,
            max_mutations=max_mutations,
            candidate_limit=200,
        )

    embeddings = load_embeddings(embeddings_path)
    surrogate = SurrogateModel.load(surrogate_dir)
    attach_surrogate_preds(surrogate, embeddings, candidates)

    if use_ram:
        if ram_module:
            ram_fn = _load_ram_callable(ram_module)
            seqs = [c.sequence for c in candidates]
            scores = ram_esm_wrapper(ram_fn, seqs)
            for c, s in zip(candidates, scores):
                c.ram_score = s
        elif ram_ref_embeddings and ram_ref_labels:
            ram_fn = build_ram_scorer(
                ref_embeddings_path=ram_ref_embeddings,
                ref_labels_path=ram_ref_labels,
                top_k=ram_top_k,
                temperature=ram_temperature,
            )
            attach_ram_scores_from_embeddings(ram_fn, candidates, embeddings)
        else:
            raise ValueError("use_ram set but no ram_module or ram reference data provided.")

    archive = QDArchive(max_mutations=max_mutations)
    batch = design_batch(
        candidates,
        archive=archive,
        batch_size=batch_size,
        beta=1.0,
        w_stability=1.0,
        w_activity=0.0,
        w_ram=1.0 if use_ram else 0.0,
        activity_floor=None,
        min_hamming=2,
    )
    return batch


def _read_fasta_first(path: Path) -> tuple[str, str]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a lightweight BO+QD round.")
    p.add_argument("--parent-fasta", type=Path, required=True, help="Parent sequence FASTA (first record used).")
    p.add_argument("--candidate-sites", required=True, help="Comma-separated 1-based positions to consider.")
    p.add_argument("--embeddings", type=Path, required=True, help="NPZ with embeddings keyed by candidate IDs.")
    p.add_argument("--surrogate-dir", type=Path, required=True, help="Directory with surrogate.pkl + meta.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-mutations", type=int, default=1)
    p.add_argument("--use-ram", action="store_true", help="Blend RAM-ESM scores if provided.")
    p.add_argument("--ram-module", type=str, default=None, help="Python module path exposing score(List[str])->np.ndarray.")
    p.add_argument("--ram-ref-embeddings", type=Path, default=None, help="Reference NPZ embeddings for RAM-ESM retrieval.")
    p.add_argument("--ram-ref-labels", type=Path, default=None, help="Reference labels CSV for RAM-ESM retrieval.")
    p.add_argument("--ram-top-k", type=int, default=5, help="Top-k neighbors for RAM scoring.")
    p.add_argument("--ram-temperature", type=float, default=0.1, help="Softmax temperature for RAM scoring.")
    p.add_argument("--use-rag-neighbors", action="store_true", help="Augment candidate set with RAG neighbor mutations.")
    p.add_argument("--rag-ref-embeddings", type=Path, default=None, help="Reference embeddings NPZ for retrieval proposals (must include parent ID).")
    p.add_argument("--rag-ref-sequences", type=Path, default=None, help="Reference sequences CSV with columns id,sequence.")
    p.add_argument("--rag-top-k", type=int, default=5, help="Top-k neighbors for proposal retrieval.")
    p.add_argument("--rag-min-similarity", type=float, default=0.5, help="Minimum cosine similarity for neighbor inclusion.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parent_id, parent_seq = _read_fasta_first(args.parent_fasta)
    sites = [int(s.strip()) for s in args.candidate_sites.split(",") if s.strip()]
    batch = run_round(
        parent_id=parent_id,
        parent_seq=parent_seq,
        candidate_sites=sites,
        embeddings_path=args.embeddings,
        surrogate_dir=args.surrogate_dir,
        batch_size=args.batch_size,
        max_mutations=args.max_mutations,
        use_ram=args.use_ram,
        ram_module=args.ram_module,
        ram_ref_embeddings=args.ram_ref_embeddings,
        ram_ref_labels=args.ram_ref_labels,
        ram_top_k=args.ram_top_k,
        ram_temperature=args.ram_temperature,
        use_rag_neighbors=args.use_rag_neighbors,
        rag_ref_embeddings=args.rag_ref_embeddings,
        rag_ref_sequences=args.rag_ref_sequences,
        rag_top_k=args.rag_top_k,
        rag_min_similarity=args.rag_min_similarity,
    )
    for cand in batch:
        print(
            f"{cand.seq_id}\tmutations={cand.mutations}\tacq={cand.acquisition}\tstab={cand.pred_stab_mean}"
        )


if __name__ == "__main__":
    main()
