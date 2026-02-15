#!/usr/bin/env python3
"""
Collect multimodel shortlist metrics (context + Rosetta + AlphaFold) into one table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from src.scoring.constraints import PROTECTED_POSITIONS
except Exception:
    PROTECTED_POSITIONS = {
        87,
        160,
        161,
        185,
        203,
        206,
        237,
        239,
        273,
        289,
    }

DISULFIDE_PAIRS = ((203, 239), (273, 289))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect AF2/Rosetta/context metrics for final shortlist.")
    p.add_argument(
        "--shortlist",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist.csv"),
        help="Final shortlist CSV from multimodel filter pipeline.",
    )
    p.add_argument(
        "--context-predictions",
        type=Path,
        default=Path("results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv"),
        help="Context model predictions CSV (for pred_abs/baseline/context metadata).",
    )
    p.add_argument(
        "--af2-dir",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/af2"),
        help="AlphaFold output directory with one subdirectory per candidate id.",
    )
    p.add_argument(
        "--triad-max",
        type=float,
        default=5.0,
        help="Maximum allowed catalytic-triad distance (angstrom) for pass flag.",
    )
    p.add_argument(
        "--disulfide-max",
        type=float,
        default=2.8,
        help="Maximum allowed disulfide SG-SG distance (angstrom) for pass flag.",
    )
    p.add_argument(
        "--plddt-min",
        type=float,
        default=90.0,
        help="pLDDT mean threshold used in pass flag.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_with_af2_metrics.csv"),
        help="Merged output table.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/surrogate_af2/multimodel_filter/final_shortlist_with_af2_summary.csv"),
        help="Summary metrics CSV.",
    )
    return p.parse_args()


def parse_pdb_atoms(path: Path) -> Dict[int, Dict[str, Tuple[float, float, float]]]:
    residues: Dict[int, Dict[str, Tuple[float, float, float]]] = {}
    if not path.exists():
        return residues
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            atom = line[12:16].strip()
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            residues.setdefault(resseq, {})[atom] = (x, y, z)
    return residues


def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _get_atom(
    res: Dict[str, Tuple[float, float, float]],
    names: Sequence[str],
) -> Optional[Tuple[float, float, float]]:
    for name in names:
        if name in res:
            return res[name]
    return None


def _min_atom_distance(
    res_a: Dict[str, Tuple[float, float, float]],
    atoms_a: Sequence[str],
    res_b: Dict[str, Tuple[float, float, float]],
    atoms_b: Sequence[str],
) -> Optional[float]:
    coords_a = [res_a[a] for a in atoms_a if a in res_a]
    coords_b = [res_b[b] for b in atoms_b if b in res_b]
    if not coords_a or not coords_b:
        return None
    return min(dist(a, b) for a in coords_a for b in coords_b)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def load_confidence_scores(path: Path) -> np.ndarray:
    data = load_json(path)
    vals = data.get("confidenceScore", [])
    try:
        arr = np.array([float(x) for x in vals], dtype=float)
    except Exception:
        return np.array([], dtype=float)
    return arr


def af2_metrics_for_id(
    rid: str,
    af2_dir: Path,
    triad_max: float,
    disulfide_max: float,
    plddt_min: float,
) -> Dict[str, object]:
    out: Dict[str, object] = {"id": rid}
    idir = af2_dir / rid

    out["af2_dir_exists"] = idir.exists() and idir.is_dir()
    out["af2_complete"] = False

    if not idir.exists() or not idir.is_dir():
        return out

    ranking = load_json(idir / "ranking_debug.json")
    if not ranking:
        return out

    order = ranking.get("order", [])
    plddts = ranking.get("plddts", {})
    if not isinstance(order, list):
        order = []
    if not isinstance(plddts, dict):
        plddts = {}

    out["af2_complete"] = True
    out["af2_ranked_model_order"] = ";".join([str(x) for x in order])
    out["af2_top_model"] = order[0] if order else None
    out["af2_models_count"] = len(plddts)

    model_means = []
    for mname, mval in plddts.items():
        try:
            model_means.append(float(mval))
        except Exception:
            continue
    if model_means:
        out["af2_model_plddt_mean_mean"] = float(np.mean(model_means))
        out["af2_model_plddt_mean_min"] = float(np.min(model_means))
        out["af2_model_plddt_mean_max"] = float(np.max(model_means))
    else:
        out["af2_model_plddt_mean_mean"] = np.nan
        out["af2_model_plddt_mean_min"] = np.nan
        out["af2_model_plddt_mean_max"] = np.nan

    top_model = out["af2_top_model"]
    if isinstance(top_model, str):
        out["af2_top_model_plddt_mean"] = float(plddts.get(top_model, np.nan))
        conf_path = idir / f"confidence_{top_model}.json"
    else:
        out["af2_top_model_plddt_mean"] = np.nan
        conf_path = Path("")

    conf_scores = load_confidence_scores(conf_path) if conf_path else np.array([], dtype=float)
    if conf_scores.size:
        key_pos = sorted(int(x) for x in PROTECTED_POSITIONS)
        key_vals = [
            conf_scores[pos - 1]
            for pos in key_pos
            if 1 <= pos <= conf_scores.size
        ]
        out["af2_ranked0_plddt_mean"] = float(np.mean(conf_scores))
        out["af2_ranked0_plddt_min"] = float(np.min(conf_scores))
        out["af2_ranked0_plddt_q25"] = float(np.percentile(conf_scores, 25))
        out["af2_ranked0_plddt_q75"] = float(np.percentile(conf_scores, 75))
        out["af2_ranked0_plddt_key_mean"] = float(np.mean(key_vals)) if key_vals else np.nan
        out["af2_ranked0_plddt_key_min"] = float(np.min(key_vals)) if key_vals else np.nan
    else:
        out["af2_ranked0_plddt_mean"] = np.nan
        out["af2_ranked0_plddt_min"] = np.nan
        out["af2_ranked0_plddt_q25"] = np.nan
        out["af2_ranked0_plddt_q75"] = np.nan
        out["af2_ranked0_plddt_key_mean"] = np.nan
        out["af2_ranked0_plddt_key_min"] = np.nan

    residues = parse_pdb_atoms(idir / "ranked_0.pdb")
    ser = residues.get(160, {})
    asp = residues.get(206, {})
    his = residues.get(237, {})
    triad_ser_his = _min_atom_distance(ser, ["OG", "CA"], his, ["NE2", "ND1", "CA"])
    triad_asp_his = _min_atom_distance(asp, ["OD2", "OD1", "CA"], his, ["NE2", "ND1", "CA"])
    triad_ser_asp = _min_atom_distance(ser, ["OG", "CA"], asp, ["OD2", "OD1", "CA"])
    out["af2_triad_ser160_his237"] = triad_ser_his
    out["af2_triad_asp206_his237"] = triad_asp_his
    out["af2_triad_ser160_asp206"] = triad_ser_asp

    triad_ok = None
    if triad_ser_his is not None and triad_asp_his is not None:
        triad_ok = bool((triad_ser_his <= triad_max) and (triad_asp_his <= triad_max))
    out["af2_triad_struct_ok"] = triad_ok

    disulfide_dists: List[Optional[float]] = []
    for i, j in DISULFIDE_PAIRS:
        cys_i = residues.get(i, {})
        cys_j = residues.get(j, {})
        sg_i = _get_atom(cys_i, ["SG"])
        sg_j = _get_atom(cys_j, ["SG"])
        disulfide_dists.append(dist(sg_i, sg_j) if sg_i and sg_j else None)
    out["af2_disulfide_203_239_dist"] = disulfide_dists[0]
    out["af2_disulfide_273_289_dist"] = disulfide_dists[1]

    disulfide_ok = None
    if all(v is not None for v in disulfide_dists):
        disulfide_ok = bool(all(v <= disulfide_max for v in disulfide_dists if v is not None))
    out["af2_disulfide_struct_ok"] = disulfide_ok

    timings = load_json(idir / "timings.json")
    if timings:
        out["af2_timing_features_s"] = float(timings.get("features", np.nan))
        pred_vals = []
        for k, v in timings.items():
            if isinstance(k, str) and k.startswith("predict_and_compile_model_"):
                try:
                    pred_vals.append(float(v))
                except Exception:
                    continue
        out["af2_timing_predict_total_s"] = float(np.sum(pred_vals)) if pred_vals else np.nan
        out["af2_timing_predict_mean_s"] = float(np.mean(pred_vals)) if pred_vals else np.nan
    else:
        out["af2_timing_features_s"] = np.nan
        out["af2_timing_predict_total_s"] = np.nan
        out["af2_timing_predict_mean_s"] = np.nan

    plddt_mean = out.get("af2_ranked0_plddt_mean", np.nan)
    plddt_key_mean = out.get("af2_ranked0_plddt_key_mean", np.nan)
    out["af2_pass_plddt_mean"] = bool(np.isfinite(plddt_mean) and plddt_mean >= plddt_min)
    out["af2_pass_plddt_key_mean"] = bool(np.isfinite(plddt_key_mean) and plddt_key_mean >= plddt_min)
    out["af2_pass_structure_geom"] = bool((triad_ok is True) and (disulfide_ok is True))
    out["af2_pass_all"] = bool(
        out["af2_pass_plddt_mean"]
        and out["af2_pass_plddt_key_mean"]
        and out["af2_pass_structure_geom"]
    )

    return out


def write_summary(path: Path, rows: List[Tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)


def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if len(vals) else float("nan")


def main() -> None:
    args = parse_args()

    shortlist = pd.read_csv(args.shortlist)
    if "id" not in shortlist.columns:
        raise SystemExit(f"Missing id column in {args.shortlist}")

    context = pd.read_csv(args.context_predictions)
    keep_cols = [c for c in ["id", "pred_abs", "baseline", "context_key", "publication"] if c in context.columns]
    context = context[keep_cols].drop_duplicates(subset=["id"], keep="first")

    merged = shortlist.merge(context, on="id", how="left")

    af2_rows = []
    for rid in merged["id"].astype(str):
        af2_rows.append(
            af2_metrics_for_id(
                rid=rid,
                af2_dir=args.af2_dir,
                triad_max=args.triad_max,
                disulfide_max=args.disulfide_max,
                plddt_min=args.plddt_min,
            )
        )
    af2_df = pd.DataFrame(af2_rows)
    merged = merged.merge(af2_df, on="id", how="left")

    if "ddg_reu" in merged.columns:
        merged["rosetta_pass_ddg_le_0"] = pd.to_numeric(merged["ddg_reu"], errors="coerce") <= 0.0
    else:
        merged["rosetta_pass_ddg_le_0"] = False

    merged["combined_structural_pass"] = (
        merged["rosetta_pass_ddg_le_0"].fillna(False)
        & merged.get("af2_pass_all", False).fillna(False)
    )

    merged = merged.sort_values("pred_residual", ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    summary_rows: List[Tuple[str, object]] = [
        ("n_shortlist", int(len(merged))),
        ("n_af2_complete", int(pd.Series(merged.get("af2_complete", False)).fillna(False).sum())),
        ("n_rosetta_pass_ddg_le_0", int(pd.Series(merged.get("rosetta_pass_ddg_le_0", False)).fillna(False).sum())),
        ("n_af2_pass_all", int(pd.Series(merged.get("af2_pass_all", False)).fillna(False).sum())),
        ("n_combined_structural_pass", int(pd.Series(merged.get("combined_structural_pass", False)).fillna(False).sum())),
        ("pred_residual_mean", safe_mean(merged.get("pred_residual", pd.Series(dtype=float)))),
        ("pred_residual_min", float(pd.to_numeric(merged.get("pred_residual", pd.Series(dtype=float)), errors="coerce").min())),
        ("pred_residual_max", float(pd.to_numeric(merged.get("pred_residual", pd.Series(dtype=float)), errors="coerce").max())),
        ("ddg_reu_mean", safe_mean(merged.get("ddg_reu", pd.Series(dtype=float)))),
        ("ddg_reu_min", float(pd.to_numeric(merged.get("ddg_reu", pd.Series(dtype=float)), errors="coerce").min())),
        ("ddg_reu_max", float(pd.to_numeric(merged.get("ddg_reu", pd.Series(dtype=float)), errors="coerce").max())),
        ("af2_ranked0_plddt_mean_mean", safe_mean(merged.get("af2_ranked0_plddt_mean", pd.Series(dtype=float)))),
        ("af2_ranked0_plddt_mean_min", float(pd.to_numeric(merged.get("af2_ranked0_plddt_mean", pd.Series(dtype=float)), errors="coerce").min())),
        ("af2_ranked0_plddt_mean_max", float(pd.to_numeric(merged.get("af2_ranked0_plddt_mean", pd.Series(dtype=float)), errors="coerce").max())),
    ]
    write_summary(args.summary_csv, summary_rows)

    print(f"Wrote merged metrics table: {args.out_csv}")
    print(f"Wrote summary table: {args.summary_csv}")


if __name__ == "__main__":
    main()

