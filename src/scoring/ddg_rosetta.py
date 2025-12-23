"""
ΔΔG helpers and Rosetta/FoldX runners.

Written to execute on a Linux host with Rosetta cartesian_ddg or FoldX installed.
No execution occurs on this macOS session; commands are composed for later use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def load_ddg_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"mutation", "ddg"}.issubset(df.columns):
        raise ValueError("ΔΔG table must contain columns: mutation, ddg")
    return df[["mutation", "ddg"]]


def estimate_ddg_from_sequence(seq: str, wt_seq: str) -> float:
    """Rough heuristic: hydropathy change per mutation."""
    if len(seq) != len(wt_seq):
        raise ValueError("Sequences must be aligned for ΔΔG estimation.")
    deltas = []
    for a, b in zip(wt_seq, seq):
        if a != b:
            deltas.append(HYDROPATHY.get(b, 0) - HYDROPATHY.get(a, 0))
    if not deltas:
        return 0.0
    return float(-np.mean(deltas))  # negative = likely stabilizing if hydropathy increases modestly


def score_sequence(
    seq: str,
    wt_seq: str,
    ddg_table: pd.DataFrame | None = None,
) -> float:
    """
    Return a stability proxy (negative ΔΔG preferred).
    Uses provided ddG table when a mutation string matches; otherwise heuristic.
    """
    if ddg_table is not None:
        muts = []
        for i, (a, b) in enumerate(zip(wt_seq, seq), start=1):
            if a != b:
                muts.append(f"{a}{i}{b}")
        if muts:
            matches = ddg_table[ddg_table["mutation"].isin(muts)]
            if not matches.empty:
                return float(matches["ddg"].mean())
    return estimate_ddg_from_sequence(seq, wt_seq)


# -------- External runners (no execution here) --------
def write_cartesian_ddg_command(
    rosetta_bin: Path,
    pdb_path: Path,
    mutations_file: Path,
    out_dir: Path,
    extra_flags: Iterable[str] | None = None,
) -> Path:
    """
    Compose a Rosetta cartesian_ddg command for later execution.
    mutations_file should contain one mutation set per line (e.g., A160S;A206D).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(rosetta_bin),
        "-s",
        str(pdb_path),
        "-ddg:mut_file",
        str(mutations_file),
        "-ddg::cartesian",
        "-beta_nov16_cart",
        "-ddg::iterations",
        "3",
        "-relax:constrain_relax_to_start_coords",
        "-relax:coord_constrain_sidechains",
        "-fa_max_dis",
        "9.0",
        "-score:weights",
        "beta_nov16_cart",
        "-out:path:all",
        str(out_dir),
    ]
    if extra_flags:
        cmd.extend(list(extra_flags))
    (out_dir / "run_cartesian_ddg.sh").write_text(" ".join(cmd))
    return out_dir / "run_cartesian_ddg.sh"


def write_foldx_command(
    foldx_bin: Path,
    pdb_path: Path,
    mutations: List[str],
    out_dir: Path,
    temperature: float = 298.15,
) -> Path:
    """
    Compose a FoldX BuildModel command for later execution.
    mutations: list like ["A160S", "A206D"].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mut_list = ",".join(mutations)
    cmd = [
        str(foldx_bin),
        "--command=BuildModel",
        f"--pdb={pdb_path.name}",
        f"--mutateResidues={mut_list}",
        f"--temperature={temperature}",
        "--numberOfRuns=3",
        f"--output-dir={out_dir}",
    ]
    (out_dir / pdb_path.name).write_bytes(pdb_path.read_bytes())
    (out_dir / "run_foldx.sh").write_text(" ".join(cmd))
    return out_dir / "run_foldx.sh"


__all__ = [
    "load_ddg_table",
    "estimate_ddg_from_sequence",
    "score_sequence",
    "write_cartesian_ddg_command",
    "write_foldx_command",
]
