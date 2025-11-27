"""ESM embedding helpers for raw amino-acid sequences.

This module keeps the runtime light until execution time (lazy imports of ``esm``/``torch``).
It is intended for small, batched embedding jobs and produces a compressed ``.npz`` mapping
sequence IDs to vectors that align with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Defaults favor a small model to avoid heavy local runs.
DEFAULT_MODEL = "esm2_t6_8M_UR50D"


def read_sequences(input_path: Path, id_col: str = "id", seq_col: str = "sequence") -> List[Tuple[str, str]]:
    """Load sequences from FASTA or CSV/TSV.

    Returns a list of (id, sequence) tuples.
    """
    suffix = input_path.suffix.lower()
    if suffix in {".fa", ".fasta"}:
        return _read_fasta(input_path)
    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        return _read_table(input_path, delim=delim, id_col=id_col, seq_col=seq_col)
    raise ValueError(f"Unsupported input format for {input_path}. Use FASTA or CSV/TSV.")


def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    seq_parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(seq_parts)))
                current_id = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    if current_id is not None:
        records.append((current_id, "".join(seq_parts)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _read_table(path: Path, delim: str, id_col: str, seq_col: str) -> List[Tuple[str, str]]:
    import csv

    records: List[Tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        missing = [c for c in (id_col, seq_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        for row in reader:
            sid = row[id_col].strip()
            seq = row[seq_col].strip()
            if sid and seq:
                records.append((sid, seq))
    if not records:
        raise ValueError(f"No sequences parsed from {path}")
    return records


def embed_sequences(
    sequences: Iterable[Tuple[str, str]],
    model_name: str = DEFAULT_MODEL,
    layer: int | None = None,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Embed sequences with an ESM model and return {id: vector}."""
    # Lazy import to keep base environment slim.
    import torch
    import esm

    model, alphabet = _load_model(model_name, device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if layer is None:
        layer = model.num_layers

    embeddings: Dict[str, np.ndarray] = {}
    seq_list = list(sequences)
    with torch.no_grad():
        for start in range(0, len(seq_list), batch_size):
            batch = seq_list[start : start + batch_size]
            data = [(sid, seq) for sid, seq in batch]
            _, batch_strs, tokens = batch_converter(data)
            tokens = tokens.to(device)
            result = model(tokens, repr_layers=[layer], return_contacts=False)
            reps = result["representations"][layer]
            for i, (sid, seq) in enumerate(batch):
                seq_len = len(seq)
                # Drop BOS/EOS and take mean-pooled token embeddings.
                token_rep = reps[i, 1 : seq_len + 1].mean(dim=0)
                embeddings[sid] = token_rep.cpu().numpy()
    return embeddings


def _load_model(model_name: str, device: str):
    import esm
    import torch

    load_fn = getattr(esm.pretrained, model_name, None)
    if load_fn is None:
        raise ValueError(f"Unknown ESM model '{model_name}'. Check esm.pretrained.* names.")
    model, alphabet = load_fn()
    model = model.to(torch.device(device))
    return model, alphabet


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed sequences with ESM and save .npz.")
    parser.add_argument("input", type=Path, help="Input FASTA or CSV/TSV with id/sequence columns.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/esm_embeddings.npz"),
        help="Output .npz path (default: data/processed/esm_embeddings.npz)",
    )
    parser.add_argument("--id-col", default="id", help="ID column for CSV/TSV inputs (default: id)")
    parser.add_argument(
        "--seq-col",
        default="sequence",
        help="Sequence column for CSV/TSV inputs (default: sequence)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ESM model name (esm.pretrained.*). Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--layer", type=int, default=None, help="Representation layer (default: top).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device string for torch (default: "cpu"). Use "cuda" if available.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = read_sequences(args.input, id_col=args.id_col, seq_col=args.seq_col)
    embeddings = embed_sequences(
        sequences,
        model_name=args.model,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_embeddings(embeddings, args.output)
    print(f"Saved {len(embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
