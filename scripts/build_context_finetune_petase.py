#!/usr/bin/env python3
"""
Build a combined dataset (FireProt context + PETase) for fine-tuning the context surrogate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine FireProt context dataset with PETase labels.")
    p.add_argument(
        "--fireprot-dataset",
        type=Path,
        default=Path("data/processed/fireprot_context_dataset.csv"),
        help="FireProt context dataset CSV.",
    )
    p.add_argument(
        "--fireprot-embeddings",
        type=Path,
        default=Path("data/processed/fireprot_context_embeddings.npz"),
        help="FireProt context embeddings NPZ.",
    )
    p.add_argument(
        "--petase-sequences",
        type=Path,
        default=Path("src/data/petase_sequences.csv"),
        help="PETase sequences CSV with id,sequence.",
    )
    p.add_argument(
        "--petase-labels",
        type=Path,
        default=Path("src/data/petase_tm_labels.csv"),
        help="PETase labels CSV with id,DTM_C (and TM_C).",
    )
    p.add_argument(
        "--petase-embeddings",
        type=Path,
        default=Path("data/processed/petase_esm_embeddings.npz"),
        help="PETase embeddings NPZ.",
    )
    p.add_argument(
        "--label-col",
        default="DTM_C",
        help="PETase label column to use (default: DTM_C).",
    )
    p.add_argument(
        "--publication",
        default="PETase_DSF",
        help="Publication/group label for PETase rows.",
    )
    p.add_argument(
        "--measure",
        default="Fluorescence",
        help="Measure label for PETase rows.",
    )
    p.add_argument(
        "--method",
        default="Thermal",
        help="Method label for PETase rows.",
    )
    p.add_argument(
        "--ph-bin",
        default="8.00",
        help="pH bin label for PETase rows (string, e.g., 8.00).",
    )
    p.add_argument(
        "--out-dataset",
        type=Path,
        default=Path("data/processed/fireprot_context_petase_finetune.csv"),
        help="Output combined dataset CSV.",
    )
    p.add_argument(
        "--out-embeddings",
        type=Path,
        default=Path("data/processed/fireprot_context_petase_embeddings.npz"),
        help="Output combined embeddings NPZ.",
    )
    return p.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def main() -> None:
    args = parse_args()
    fire = pd.read_csv(args.fireprot_dataset)
    fire = fire[fire["label_type"] == "DTM"].copy()

    pet_seq = pd.read_csv(args.petase_sequences)
    pet_lab = pd.read_csv(args.petase_labels)
    if args.label_col not in pet_lab.columns:
        raise SystemExit(f"Missing {args.label_col} in {args.petase_labels}")

    pet = pet_seq.merge(pet_lab, on="id", how="inner")
    pet = pet[["id", "sequence", args.label_col]].rename(columns={args.label_col: "label"})
    pet["label_type"] = "DTM"
    pet["publication"] = args.publication
    pet["measure"] = args.measure
    pet["method"] = args.method
    pet["ph_bin"] = args.ph_bin
    pet["context_key"] = (
        pet["publication"] + "|" + pet["measure"] + "|" + pet["method"] + "|" + pet["ph_bin"]
    )

    keep_cols = ["id", "label", "label_type", "context_key", "publication", "measure", "method", "ph_bin"]
    fire_out = fire[keep_cols].copy()
    pet_out = pet[keep_cols].copy()

    combined = pd.concat([fire_out, pet_out], ignore_index=True)

    fire_emb = load_npz(args.fireprot_embeddings)
    pet_emb = load_npz(args.petase_embeddings)

    # Filter to IDs with embeddings
    combined = combined[combined["id"].isin(set(fire_emb) | set(pet_emb))].copy()

    merged = {}
    merged.update(fire_emb)
    merged.update(pet_emb)

    args.out_dataset.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_dataset, index=False)
    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_embeddings, **merged)

    print(f"Wrote {args.out_dataset} with {len(combined)} rows")
    print(f"Wrote {args.out_embeddings} with {len(merged)} embeddings")


if __name__ == "__main__":
    main()
