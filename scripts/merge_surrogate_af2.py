#!/usr/bin/env python3
"""
Merge surrogate predictions with AF2 metrics into a single CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

from src.scoring.af2_interface import load_af2_metrics


def _extract_id_from_json(path: Path) -> Optional[str]:
    name = path.name
    if name == "config.json":
        return None
    if "_scores_" in name:
        return name.split("_scores_")[0]
    if name.endswith("_scores.json"):
        return name[: -len("_scores.json")]
    # fallback: use stem
    return path.stem


def _load_predictions(path: Path) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"Missing id column in {path}")
        if "prediction" not in reader.fieldnames:
            raise ValueError(f"Missing prediction column in {path}")
        for row in reader:
            rid = row.get("id", "").strip()
            val = row.get("prediction", "").strip()
            if not rid or not val:
                continue
            try:
                preds[rid] = float(val)
            except ValueError:
                continue
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge surrogate predictions with AF2 metrics.")
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/predictions.csv"),
        help="Surrogate predictions CSV.",
    )
    p.add_argument(
        "--af2-dir",
        type=Path,
        default=Path("results/surrogate_af2/af2"),
        help="AF2 output directory.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/surrogate_af2/predictions_with_af2.csv"),
        help="Output merged CSV.",
    )
    p.add_argument("--plddt-mean-floor", type=float, default=0.0)
    p.add_argument("--plddt-min-floor", type=float, default=0.0)
    p.add_argument("--pae-max", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preds = _load_predictions(args.predictions)
    if not preds:
        raise SystemExit(f"No predictions found in {args.predictions}")

    rows = []
    for json_path in args.af2_dir.rglob("*.json"):
        seq_id = _extract_id_from_json(json_path)
        if not seq_id:
            continue
        metrics = load_af2_metrics(json_path)
        rows.append(
            {
                "id": seq_id,
                "prediction": preds.get(seq_id),
                "plddt_mean": metrics.get("plddt_mean", 0.0),
                "plddt_min": metrics.get("plddt_min", 0.0),
                "pae_mean": metrics.get("pae_mean", 0.0),
            }
        )

    rows = [r for r in rows if r["prediction"] is not None]
    if not rows:
        raise SystemExit("No overlapping IDs between predictions and AF2 metrics.")

    for row in rows:
        passes = row["plddt_mean"] >= args.plddt_mean_floor
        passes = passes and row["plddt_min"] >= args.plddt_min_floor
        if args.pae_max is not None:
            passes = passes and row["pae_mean"] <= args.pae_max
        row["passes_filter"] = passes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "prediction",
                "plddt_mean",
                "plddt_min",
                "pae_mean",
                "passes_filter",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
