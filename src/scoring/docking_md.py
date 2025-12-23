"""
Docking/MD helpers and command composers.

Intended for Linux with GNINA/AutoDock Vina and optional OpenMM; nothing runs on macOS here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_docking_scores(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    if not {"seq_id", "docking_score"}.issubset(df.columns):
        raise ValueError("Docking table must contain seq_id,docking_score.")
    return dict(zip(df["seq_id"], df["docking_score"]))


def load_md_contacts(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    if not {"seq_id", "contact_fraction"}.issubset(df.columns):
        raise ValueError("MD contacts table must contain seq_id,contact_fraction.")
    return dict(zip(df["seq_id"], df["contact_fraction"]))


def activity_proxy(
    seq_id: str, docking: Dict[str, float], contacts: Dict[str, float] | None = None
) -> float:
    dock = docking.get(seq_id)
    if dock is None:
        return 0.0
    contact = contacts.get(seq_id, 0.0) if contacts else 0.0
    # Lower (better) docking score mapped to higher activity; boost with contacts
    return float(-dock + 2.0 * contact)


def write_gnina_command(
    gnina_bin: Path,
    receptor: Path,
    ligand: Path,
    out_path: Path,
    center: Optional[List[float]] = None,
    size: Optional[List[float]] = None,
    exhaustiveness: int = 8,
) -> Path:
    """
    Compose a GNINA/AutoDock-Vina command for later execution.
    receptor/ligand should be pdbqt files; center/size define the box.
    """
    cmd = [
        str(gnina_bin),
        "-r",
        str(receptor),
        "-l",
        str(ligand),
        "-o",
        str(out_path),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "9",
    ]
    if center and size:
        cmd.extend(
            [
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / "run_gnina.sh").write_text(" ".join(cmd))
    return out_path.parent / "run_gnina.sh"


__all__ = ["load_docking_scores", "load_md_contacts", "activity_proxy", "write_gnina_command"]
