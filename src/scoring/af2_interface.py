"""
AlphaFold2/ColabFold runner and result reader.

This module is written for a Linux + CUDA host where AF2/ColabFold is installed.
It does not execute on macOS; we only compose CLI commands and parse outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def run_colabfold(
    fasta_path: Path,
    out_dir: Path,
    binary: str = "colabfold_batch",
    model_preset: str = "monomer",
    use_templates: bool = False,
    num_models: int = 1,
    num_recycles: int = 3,
    amber_relax: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Launch ColabFold/AF2 for the sequences in fasta_path, writing outputs to out_dir.
    Returns the output directory path. Does not run on macOS in this session.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "--model-type",
        model_preset,
        "--num-models",
        str(num_models),
        "--num-recycles",
        str(num_recycles),
        "--output-dir",
        str(out_dir),
    ]
    if use_templates:
        cmd.append("--templates")
    if not amber_relax:
        cmd.extend(["--amber", "off"])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(fasta_path))
    # Do not run here; user runs on Linux/GPU. Provide command for later execution.
    (out_dir / "colabfold_command.txt").write_text(" ".join(cmd))
    return out_dir


def find_af2_json(out_dir: Path) -> Optional[Path]:
    """Find the first AF2 JSON sidecar in an output directory."""
    for p in out_dir.glob("*.json"):
        return p
    subdirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    for sd in subdirs:
        for p in sd.glob("*.json"):
            return p
    return None


def load_af2_metrics(json_path: Path) -> Dict[str, float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    plddt = np.array(data.get("plddt", []), dtype=float)
    pae = np.array(data.get("pae", []), dtype=float)
    metrics = {}
    if plddt.size:
        metrics["plddt_mean"] = float(plddt.mean())
        metrics["plddt_min"] = float(plddt.min())
    if pae.size:
        metrics["pae_mean"] = float(pae.mean())
    if "active_site_rmsd" in data:
        metrics["active_site_rmsd"] = float(data["active_site_rmsd"])
    return metrics


def passes_structure_checks(
    metrics: Dict[str, float], plddt_floor: float = 70.0, max_rmsd: float = 2.5
) -> bool:
    if metrics.get("plddt_mean", 0.0) < plddt_floor:
        return False
    if metrics.get("active_site_rmsd") is not None and metrics["active_site_rmsd"] > max_rmsd:
        return False
    return True


__all__ = ["run_colabfold", "find_af2_json", "load_af2_metrics", "passes_structure_checks"]
