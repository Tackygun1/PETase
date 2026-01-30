#!/usr/bin/env python3
"""
Build a PETase similarity pretrain set using ESM embeddings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src.acquisition.retrieval import load_ref_embeddings, normalize


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _read_label_map(path: Path, id_col: str, label_col: str) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [c for c in (id_col, label_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        for row in reader:
            rid = row.get(id_col, "").strip()
            raw = row.get(label_col, "").strip()
            if not rid or not raw:
                continue
            try:
                labels[rid] = float(raw)
            except ValueError:
                continue
    return labels


def _write_labels(path: Path, rows: Iterable[Tuple[str, float]], target_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", target_col])
        writer.writeheader()
        for rid, val in rows:
            writer.writerow({"id": rid, target_col: val})


def _select_similar(
    ref_ids: list[str],
    ref_mat: np.ndarray,
    query: np.ndarray,
    top_k: int,
    min_sim: float | None,
) -> Tuple[list[str], list[float]]:
    q = query / (np.linalg.norm(query) + 1e-8)
    sims = ref_mat @ q
    if min_sim is not None:
        idx = np.where(sims >= min_sim)[0]
        idx = idx[np.argsort(sims[idx])[::-1]]
    else:
        idx = np.argsort(sims)[::-1][:top_k]
    selected_ids = [ref_ids[i] for i in idx]
    selected_sims = [float(sims[i]) for i in idx]
    return selected_ids, selected_sims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PETase similarity pretrain set from embeddings + labels."
    )
    parser.add_argument(
        "--ref-embeddings",
        type=Path,
        default=Path("data/processed/esm_embeddings.npz"),
        help="Reference embeddings (e.g., FireProt) .npz.",
    )
    parser.add_argument(
        "--ref-labels",
        type=Path,
        default=Path("data/processed/labels.csv"),
        help="Reference labels CSV.",
    )
    parser.add_argument("--ref-id-col", default="id", help="ID column in ref labels.")
    parser.add_argument(
        "--ref-label-col",
        default="stability",
        help="Label column in ref labels (default: stability).",
    )
    parser.add_argument(
        "--petase-embeddings",
        type=Path,
        default=Path("data/processed/petase_esm_embeddings.npz"),
        help="PETase embeddings .npz.",
    )
    parser.add_argument(
        "--petase-labels",
        type=Path,
        default=Path("src/data/petase_tm_labels.csv"),
        help="PETase labels CSV.",
    )
    parser.add_argument("--petase-id-col", default="id", help="ID column in PETase labels.")
    parser.add_argument(
        "--petase-label-col",
        default="TM_C",
        help="PETase label column (default: TM_C).",
    )
    parser.add_argument(
        "--target-col",
        default="stability",
        help="Output label column name (default: stability).",
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="Use a specific PETase ID as the similarity query (default: mean embedding).",
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-K similar refs to keep.")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=None,
        help="Optional cosine similarity threshold (overrides top-k).",
    )
    parser.add_argument(
        "--out-embeddings",
        type=Path,
        default=Path("data/processed/petase_pretrain_embeddings.npz"),
        help="Output combined embeddings .npz.",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=Path("data/processed/petase_pretrain_labels.csv"),
        help="Output combined labels CSV.",
    )
    parser.add_argument(
        "--out-ids",
        type=Path,
        default=None,
        help="Optional path to save selected ref IDs with similarity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_mat, ref_ids = load_ref_embeddings(args.ref_embeddings)
    ref_mat = normalize(ref_mat)
    petase_embs = _load_npz(args.petase_embeddings)
    if not petase_embs:
        raise SystemExit("No PETase embeddings found.")

    if args.query_id:
        if args.query_id not in petase_embs:
            raise SystemExit(f"Query id {args.query_id} not found in PETase embeddings.")
        query = petase_embs[args.query_id]
    else:
        query = np.mean(np.stack(list(petase_embs.values()), axis=0), axis=0)

    if args.min_sim is None and args.top_k <= 0:
        raise SystemExit("Set --top-k > 0 or provide --min-sim.")

    selected_ids, selected_sims = _select_similar(
        ref_ids, ref_mat, query, top_k=args.top_k, min_sim=args.min_sim
    )

    ref_labels = _read_label_map(args.ref_labels, args.ref_id_col, args.ref_label_col)
    petase_labels = _read_label_map(
        args.petase_labels, args.petase_id_col, args.petase_label_col
    )

    ref_data = np.load(args.ref_embeddings, allow_pickle=False)
    out_embeddings: Dict[str, np.ndarray] = {}
    kept_ref = 0
    for rid in selected_ids:
        if rid in ref_data:
            out_embeddings[rid] = ref_data[rid]
            kept_ref += 1
    for pid, vec in petase_embs.items():
        out_embeddings[pid] = vec

    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_embeddings, **out_embeddings)

    label_rows: list[Tuple[str, float]] = []
    missing_ref = 0
    for rid in selected_ids:
        if rid in ref_labels:
            label_rows.append((rid, ref_labels[rid]))
        else:
            missing_ref += 1
    for pid, val in petase_labels.items():
        label_rows.append((pid, val))
    _write_labels(args.out_labels, label_rows, args.target_col)

    if args.out_ids:
        args.out_ids.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_ids, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "similarity"])
            writer.writeheader()
            for rid, sim_val in zip(selected_ids, selected_sims):
                writer.writerow({"id": rid, "similarity": sim_val})

    print(f"Selected {len(selected_ids)} similar refs (kept {kept_ref} with embeddings).")
    print(f"Ref labels found: {len(selected_ids) - missing_ref}, missing: {missing_ref}.")
    print(f"PETase labels: {len(petase_labels)}.")
    print(f"Wrote embeddings: {args.out_embeddings}")
    print(f"Wrote labels: {args.out_labels}")


if __name__ == "__main__":
    main()
