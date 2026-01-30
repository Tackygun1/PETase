#!/usr/bin/env python3
"""
QC checks for generated PETase candidates.

Sequence-level:
- length, mutation count vs WT
- protected positions unchanged
- disulfide cysteines present
- N-X-S/T motif (glycosylation risk)
- duplicates / training-set leakage

Structure-level (AF2/ColabFold):
- pLDDT mean/min + key-site pLDDT
- catalytic triad distances
- disulfide S-S distances
- optional RMSD to WT structure
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from src.scoring.constraints import PROTECTED_POSITIONS
except Exception:
    PROTECTED_POSITIONS = {
        160,
        206,
        237,
        87,
        161,
        185,
        203,
        239,
        273,
        289,
    }

CATALYTIC_TRIAD = (160, 206, 237)
DISULFIDE_PAIRS = ((203, 239), (273, 289))
KEY_POSITIONS = sorted(PROTECTED_POSITIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC checks for generated candidates.")
    p.add_argument(
        "--candidates-fasta",
        type=Path,
        default=Path("data/processed/candidates.fasta"),
        help="FASTA with generated candidates.",
    )
    p.add_argument(
        "--wt-fasta",
        type=Path,
        default=Path("data/processed/petase_wt.fasta"),
        help="FASTA containing the WT sequence.",
    )
    p.add_argument(
        "--train-seq-db",
        type=Path,
        default=Path("data/processed/sequence_db_petase_pretrain.csv"),
        help="CSV with training sequences (id,sequence).",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/surrogate_af2/predictions.csv"),
        help="Surrogate predictions CSV (id,prediction).",
    )
    p.add_argument(
        "--af2-dir",
        type=Path,
        default=Path("results/surrogate_af2/af2"),
        help="AF2 output directory (for pLDDT/PDB checks).",
    )
    p.add_argument(
        "--wt-pdb",
        type=Path,
        default=None,
        help="WT PDB to compute RMSD against (optional).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/surrogate_af2/qc_report.csv"),
        help="Output QC CSV.",
    )
    p.add_argument("--plddt-mean-min", type=float, default=70.0)
    p.add_argument("--plddt-key-min", type=float, default=70.0)
    p.add_argument("--max-disulfide-dist", type=float, default=2.8)
    p.add_argument("--max-triad-dist", type=float, default=5.0)
    return p.parse_args()


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    if not path.exists():
        return seqs
    current_id = None
    parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(parts)
                current_id = line[1:].strip().split()[0]
                parts = []
            else:
                parts.append(line)
        if current_id:
            seqs[current_id] = "".join(parts)
    return seqs


def load_predictions(path: Path) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    if not path.exists():
        return preds
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return preds
        if "id" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            return preds
        for row in reader:
            rid = (row.get("id") or "").strip()
            val = (row.get("prediction") or "").strip()
            if not rid or not val:
                continue
            try:
                preds[rid] = float(val)
            except ValueError:
                continue
    return preds


def load_train_sequences(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    if not path.exists():
        return seqs
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return seqs
        if "id" not in reader.fieldnames or "sequence" not in reader.fieldnames:
            return seqs
        for row in reader:
            rid = (row.get("id") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if rid and seq:
                seqs[rid] = seq
    return seqs


def glyco_motif_count(seq: str) -> int:
    if len(seq) < 3:
        return 0
    count = 0
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 2] in ("S", "T"):
            count += 1
    return count


def seq_mutations(seq: str, wt: str) -> List[str]:
    muts = []
    for i, (a, b) in enumerate(zip(wt, seq), start=1):
        if a != b:
            muts.append(f"{a}{i}{b}")
    return muts


def extract_prefix(name: str) -> Optional[str]:
    if "_scores_" in name:
        return name.split("_scores_")[0]
    if "_unrelaxed_" in name:
        return name.split("_unrelaxed_")[0]
    if "_relaxed_" in name:
        return name.split("_relaxed_")[0]
    if name.endswith("_scores.json"):
        return name[: -len("_scores.json")]
    return None


def index_af2_outputs(af2_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    pdbs: Dict[str, Path] = {}
    jsons: Dict[str, Path] = {}
    if not af2_dir.exists():
        return pdbs, jsons
    for p in af2_dir.iterdir():
        if p.is_dir():
            continue
        if p.suffix == ".pdb":
            prefix = extract_prefix(p.name)
            if prefix:
                pdbs[prefix] = p
        if p.suffix == ".json":
            prefix = extract_prefix(p.name)
            if prefix:
                jsons[prefix] = p
    return pdbs, jsons


def load_plddt(json_path: Path) -> List[float]:
    if not json_path.exists():
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    plddt = data.get("plddt", [])
    return [float(x) for x in plddt]


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
            except ValueError:
                continue
            try:
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


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    C = P.T @ Q
    V, _, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ W))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ W


def rmsd_align(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    P_center = P - P.mean(axis=0)
    Q_center = Q - Q.mean(axis=0)
    R = kabsch_align(P_center, Q_center)
    P_rot = P_center @ R
    diff = P_rot - Q_center
    rmsd = math.sqrt((diff * diff).sum() / len(P_rot))
    return rmsd, (P_rot + Q.mean(axis=0))


def collect_ca_coords(
    residues: Dict[int, Dict[str, Tuple[float, float, float]]],
    positions: Iterable[int],
) -> Dict[int, Tuple[float, float, float]]:
    coords = {}
    for pos in positions:
        atom = residues.get(pos, {}).get("CA")
        if atom:
            coords[pos] = atom
    return coords


def main() -> None:
    args = parse_args()
    candidates = read_fasta(args.candidates_fasta)
    if not candidates:
        raise SystemExit(f"No candidates found in {args.candidates_fasta}")

    wt_seqs = read_fasta(args.wt_fasta)
    wt_seq = None
    if wt_seqs:
        wt_seq = next(iter(wt_seqs.values()))

    preds = load_predictions(args.predictions)
    train_seqs = load_train_sequences(args.train_seq_db)
    train_seq_set = set(train_seqs.values())

    seq_to_first_id: Dict[str, str] = {}
    duplicate_ids: Dict[str, bool] = {}
    for cid, seq in candidates.items():
        if seq in seq_to_first_id:
            duplicate_ids[cid] = True
        else:
            seq_to_first_id[seq] = cid
            duplicate_ids[cid] = False

    pdbs, jsons = index_af2_outputs(args.af2_dir)
    wt_residues = parse_pdb_atoms(args.wt_pdb) if args.wt_pdb else {}
    wt_ca = collect_ca_coords(wt_residues, range(1, len(wt_seq) + 1)) if wt_seq and wt_residues else {}

    rows = []
    for cid, seq in candidates.items():
        row: Dict[str, object] = {"id": cid}
        row["prediction"] = preds.get(cid)
        row["length"] = len(seq)
        row["in_training_seq"] = seq in train_seq_set if train_seq_set else None
        row["duplicate_sequence"] = duplicate_ids.get(cid, False)
        wt_glyco = glyco_motif_count(wt_seq) if wt_seq else 0
        row["glyco_motif_new"] = glyco_motif_count(seq) > wt_glyco

        protected_mutations = []
        mut_count = None
        if wt_seq and len(wt_seq) == len(seq):
            muts = seq_mutations(seq, wt_seq)
            mut_count = len(muts)
            for mut in muts:
                pos = int(mut[1:-1])
                if pos in PROTECTED_POSITIONS:
                    protected_mutations.append(mut)
        row["mut_count"] = mut_count
        row["protected_mutations"] = ";".join(protected_mutations)

        disulfide_seq_ok = True
        for i, j in DISULFIDE_PAIRS:
            if i <= len(seq) and j <= len(seq):
                disulfide_seq_ok = disulfide_seq_ok and seq[i - 1] == "C" and seq[j - 1] == "C"
        row["disulfide_seq_ok"] = disulfide_seq_ok

        plddt = []
        pdb_path = pdbs.get(cid)
        json_path = jsons.get(cid)
        if json_path:
            plddt = load_plddt(json_path)
        if plddt:
            row["plddt_mean"] = float(sum(plddt) / len(plddt))
            row["plddt_min"] = float(min(plddt))
            key_vals = [plddt[pos - 1] for pos in KEY_POSITIONS if pos - 1 < len(plddt)]
            row["plddt_key_mean"] = float(sum(key_vals) / len(key_vals)) if key_vals else None
            row["plddt_key_min"] = float(min(key_vals)) if key_vals else None
        else:
            row["plddt_mean"] = None
            row["plddt_min"] = None
            row["plddt_key_mean"] = None
            row["plddt_key_min"] = None

        residues = parse_pdb_atoms(pdb_path) if pdb_path else {}
        # Distances: catalytic triad and disulfides
        ser = residues.get(160, {})
        asp = residues.get(206, {})
        his = residues.get(237, {})
        row["triad_ser160_his237"] = _min_atom_distance(
            ser, ["OG", "CA"], his, ["NE2", "ND1", "CA"]
        )
        row["triad_asp206_his237"] = _min_atom_distance(
            asp, ["OD2", "OD1", "CA"], his, ["NE2", "ND1", "CA"]
        )
        row["triad_ser160_asp206"] = _min_atom_distance(
            ser, ["OG", "CA"], asp, ["OD2", "OD1", "CA"]
        )

        disulfide_dists = []
        for i, j in DISULFIDE_PAIRS:
            cys_i = residues.get(i, {})
            cys_j = residues.get(j, {})
            sg_i = _get_atom(cys_i, ["SG"])
            sg_j = _get_atom(cys_j, ["SG"])
            disulfide_dists.append(dist(sg_i, sg_j) if sg_i and sg_j else None)
        row["disulfide_203_239_dist"] = disulfide_dists[0]
        row["disulfide_273_289_dist"] = disulfide_dists[1]

        triad_ok = True
        for key in ("triad_ser160_his237", "triad_asp206_his237"):
            val = row.get(key)
            if val is not None and val > args.max_triad_dist:
                triad_ok = False
        row["triad_struct_ok"] = triad_ok if residues else None

        disulfide_ok = True
        for val in disulfide_dists:
            if val is not None and val > args.max_disulfide_dist:
                disulfide_ok = False
        row["disulfide_struct_ok"] = disulfide_ok if residues else None

        # Optional RMSD to WT structure
        if residues and wt_ca:
            ca_coords = collect_ca_coords(residues, wt_ca.keys())
            common = sorted(set(ca_coords.keys()) & set(wt_ca.keys()))
            if common:
                P = np.array([ca_coords[pos] for pos in common], dtype=float)
                Q = np.array([wt_ca[pos] for pos in common], dtype=float)
                rmsd_all, _ = rmsd_align(P, Q)
                row["rmsd_ca"] = rmsd_all
                key_common = sorted(set(common) & set(KEY_POSITIONS))
                if key_common:
                    Pk = np.array([ca_coords[pos] for pos in key_common], dtype=float)
                    Qk = np.array([wt_ca[pos] for pos in key_common], dtype=float)
                    rmsd_key, _ = rmsd_align(Pk, Qk)
                    row["rmsd_key_ca"] = rmsd_key
                else:
                    row["rmsd_key_ca"] = None
            else:
                row["rmsd_ca"] = None
                row["rmsd_key_ca"] = None
        else:
            row["rmsd_ca"] = None
            row["rmsd_key_ca"] = None

        flags = []
        if protected_mutations:
            flags.append("protected_mutations")
        if row["disulfide_seq_ok"] is False:
            flags.append("disulfide_seq")
        if row["glyco_motif_new"]:
            flags.append("glyco_motif_new")
        if row["duplicate_sequence"]:
            flags.append("duplicate_seq")
        if row["in_training_seq"]:
            flags.append("train_leakage")
        if row["plddt_mean"] is not None and row["plddt_mean"] < args.plddt_mean_min:
            flags.append("low_plddt_mean")
        if row["plddt_key_min"] is not None and row["plddt_key_min"] < args.plddt_key_min:
            flags.append("low_plddt_key")
        if row["disulfide_struct_ok"] is False:
            flags.append("disulfide_geom")
        if row["triad_struct_ok"] is False:
            flags.append("triad_geom")
        row["flags"] = ";".join(flags)

        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "prediction",
        "length",
        "mut_count",
        "protected_mutations",
        "disulfide_seq_ok",
        "glyco_motif_new",
        "duplicate_sequence",
        "in_training_seq",
        "plddt_mean",
        "plddt_min",
        "plddt_key_mean",
        "plddt_key_min",
        "triad_ser160_his237",
        "triad_asp206_his237",
        "triad_ser160_asp206",
        "triad_struct_ok",
        "disulfide_203_239_dist",
        "disulfide_273_289_dist",
        "disulfide_struct_ok",
        "rmsd_ca",
        "rmsd_key_ca",
        "flags",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote QC report to {args.out} ({len(rows)} sequences).")


if __name__ == "__main__":
    main()
