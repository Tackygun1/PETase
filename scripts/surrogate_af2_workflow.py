#!/usr/bin/env python3
"""
Rank candidates with an XGBoost surrogate, then prepare an AF2/ColabFold run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
import shlex
import shutil
import subprocess

from src.scoring.af2_interface import run_colabfold


def read_candidates(
    path: Path, id_col: str = "id", seq_col: str = "sequence"
) -> Dict[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta"}:
        return _read_fasta(path)
    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        return _read_table(path, id_col=id_col, seq_col=seq_col, delim=delim)
    raise ValueError(f"Unsupported input format: {path}")


def _read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_id = None
    seq_parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(seq_parts)
                current_id = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    if current_id is not None:
        records[current_id] = "".join(seq_parts)
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _read_table(path: Path, id_col: str, seq_col: str, delim: str) -> Dict[str, str]:
    records: Dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [c for c in (id_col, seq_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        for row in reader:
            rid = row.get(id_col, "").strip()
            seq = row.get(seq_col, "").strip()
            if rid and seq:
                records[rid] = seq
    if not records:
        raise ValueError(f"No sequences parsed from {path}")
    return records


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def write_fasta(records: Iterable[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rid, seq in records:
            f.write(f">{rid}\n")
            f.write(f"{seq}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score candidates, then stage AF2 inputs.")
    p.add_argument("--candidates", type=Path, required=True, help="FASTA or CSV with id/sequence.")
    p.add_argument("--embeddings", type=Path, required=True, help="Embeddings NPZ for candidates.")
    p.add_argument("--model", type=Path, required=True, help="XGBoost model JSON.")
    p.add_argument("--id-col", default="id", help="ID column for CSV inputs.")
    p.add_argument("--seq-col", default="sequence", help="Sequence column for CSV inputs.")
    p.add_argument("--top-k", type=int, default=20, help="Top K sequences to keep.")
    p.add_argument(
        "--direction",
        choices=("max", "min"),
        default="max",
        help="Sort direction for scores (max=larger is better).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("results/surrogate_af2"))
    p.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional path for predictions CSV (default: out-dir/predictions.csv).",
    )
    p.add_argument(
        "--scoring-config",
        type=Path,
        default=Path("configs/scoring_config.yaml"),
        help="YAML with AF2 settings.",
    )
    p.add_argument(
        "--af2-dir",
        type=Path,
        default=None,
        help="Override AF2 output directory (default: out-dir/af2).",
    )
    p.add_argument(
        "--af2-bin",
        type=str,
        default=None,
        help="Override ColabFold binary (otherwise read from scoring_config.yaml).",
    )
    p.add_argument(
        "--run-af2",
        action="store_true",
        help="Execute the ColabFold command after writing it.",
    )
    p.add_argument("--skip-af2", action="store_true", help="Skip AF2 command generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    candidates = read_candidates(args.candidates, id_col=args.id_col, seq_col=args.seq_col)
    embeddings = load_embeddings(args.embeddings)
    ids = [rid for rid in candidates.keys() if rid in embeddings]
    if not ids:
        raise SystemExit("No candidate IDs found in embeddings.")

    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(args.model)

    X = np.vstack([embeddings[rid] for rid in ids])
    preds = model.predict(X)

    scored = list(zip(ids, preds))
    reverse = args.direction == "max"
    scored.sort(key=lambda x: x[1], reverse=reverse)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.predictions_out or out_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        for rid, score in scored:
            writer.writerow({"id": rid, "prediction": f"{score:.6f}"})

    top_k = args.top_k if args.top_k > 0 else len(scored)
    top_ids = [rid for rid, _ in scored[:top_k]]
    fasta_path = out_dir / "topk.fasta"
    write_fasta([(rid, candidates[rid]) for rid in top_ids], fasta_path)

    print(f"Wrote predictions to {pred_path}")
    print(f"Wrote top-{top_k} FASTA to {fasta_path}")

    if args.skip_af2:
        return

    af2_cfg = {}
    if args.scoring_config.exists():
        af2_cfg = (yaml.safe_load(args.scoring_config.read_text()) or {}).get("af2", {})

    out_af2 = args.af2_dir or (out_dir / "af2")
    model_type = str(af2_cfg.get("model_type", af2_cfg.get("model_preset", "auto")))
    if model_type == "monomer":
        model_type = "alphafold2"
    elif model_type == "multimer":
        model_type = "alphafold2_multimer_v3"
    af2_bin = args.af2_bin or str(af2_cfg.get("binary", "colabfold_batch"))
    run_colabfold(
        fasta_path=fasta_path,
        out_dir=out_af2,
        binary=af2_bin,
        model_preset=model_type,
        use_templates=bool(af2_cfg.get("use_templates", False)),
        num_models=int(af2_cfg.get("num_models", 1)),
        num_recycles=int(af2_cfg.get("num_recycles", 3)),
        amber_relax=bool(af2_cfg.get("amber_relax", False)),
    )
    print(f"Wrote ColabFold command to {out_af2 / 'colabfold_command.txt'}")
    if args.run_af2:
        cmd_path = out_af2 / "colabfold_command.txt"
        cmd_text = cmd_path.read_text().strip()
        if not cmd_text:
            raise SystemExit("ColabFold command file is empty.")
        cmd_parts = shlex.split(cmd_text)
        binary = cmd_parts[0]
        if not (Path(binary).exists() or shutil.which(binary)):
            raise SystemExit(f"ColabFold binary not found: {binary}")
        subprocess.run(cmd_parts, check=True)


if __name__ == "__main__":
    main()
