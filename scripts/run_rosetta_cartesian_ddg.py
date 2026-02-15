#!/usr/bin/env python3
"""
Run Rosetta cartesian_ddg for a set of mutfiles and collect ΔΔG scores.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Rosetta cartesian_ddg in batch.")
    p.add_argument("--rosetta-bin", type=Path, required=True, help="cartesian_ddg binary.")
    p.add_argument("--wt-pdb", type=Path, required=True, help="WT PDB path.")
    p.add_argument(
        "--mutfiles-dir",
        type=Path,
        required=True,
        help="Directory containing mutfiles.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/rosetta/ddg_runs"),
        help="Output directory for Rosetta outputs.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/rosetta/ddg_scores.csv"),
        help="Output CSV with id, ddg.",
    )
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel Rosetta jobs.",
    )
    p.add_argument(
        "--extra-flags",
        default="",
        help="Extra Rosetta flags as a single string.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip mutfiles that already have a parsable scorefile.",
    )
    return p.parse_args()


def parse_scorefile(path: Path) -> float | None:
    if not path.exists():
        return None
    headers = None
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("SCORE:"):
                parts = line.strip().split()
                if headers is None:
                    headers = parts[1:]
                else:
                    rows.append(parts[1:])
    if not headers or not rows:
        return None
    # Try ddg column, fallback to total_score
    col = None
    for name in ("ddg", "total_score"):
        if name in headers:
            col = headers.index(name)
            break
    if col is None:
        return None
    # Use last row
    try:
        return float(rows[-1][col])
    except Exception:
        return None


def parse_ddgfile(path: Path) -> float | None:
    if not path.exists():
        return None
    wt_vals: List[float] = []
    mut_vals: List[float] = []
    with open(path, "r") as f:
        for line in f:
            if " WT:" in line:
                parts = line.strip().split()
                try:
                    idx = parts.index("WT:")
                    wt_vals.append(float(parts[idx + 1]))
                except Exception:
                    continue
            elif " MUT_" in line:
                parts = line.strip().split()
                try:
                    idx = parts.index(next(p for p in parts if p.startswith("MUT_")))
                    mut_vals.append(float(parts[idx + 1]))
                except Exception:
                    continue
    if not wt_vals or not mut_vals:
        return None
    return (sum(mut_vals) / len(mut_vals)) - (sum(wt_vals) / len(wt_vals))


def _run_one(mutfile: Path, args: argparse.Namespace, rosetta: Path) -> Dict[str, object]:
    cid = mutfile.stem
    out_sub = args.out_dir / cid
    out_sub.mkdir(parents=True, exist_ok=True)
    scorefile = out_sub / "score.sc"
    ddgfile = out_sub / f"{cid}.ddg"
    ddgfile_cwd = Path(f"{cid}.ddg")

    if args.resume and scorefile.exists():
        ddg = parse_scorefile(scorefile)
        if ddg is not None:
            return {"id": cid, "ddg": ddg, "scorefile": str(scorefile)}
    if args.resume:
        ddg = parse_ddgfile(ddgfile) or parse_ddgfile(ddgfile_cwd)
        if ddg is not None:
            return {"id": cid, "ddg": ddg, "scorefile": str(ddgfile if ddgfile.exists() else ddgfile_cwd)}

    cmd = [
        str(rosetta),
        "-s",
        str(args.wt_pdb),
        "-ddg:mut_file",
        str(mutfile),
        "-ddg::cartesian",
        "-beta_nov16_cart",
        "-ddg::iterations",
        str(args.iterations),
        "-relax:constrain_relax_to_start_coords",
        "-relax:coord_constrain_sidechains",
        "-fa_max_dis",
        "9.0",
        "-score:weights",
        "beta_nov16_cart",
        "-out:path:all",
        str(out_sub),
        "-out:file:scorefile",
        str(scorefile),
    ]
    if args.extra_flags.strip():
        cmd.extend(args.extra_flags.split())

    subprocess.run(cmd, check=True)
    ddg = parse_scorefile(scorefile)
    if ddg is None:
        ddg = parse_ddgfile(ddgfile) or parse_ddgfile(ddgfile_cwd)
        return {"id": cid, "ddg": ddg, "scorefile": str(ddgfile if ddgfile.exists() else ddgfile_cwd)}
    return {"id": cid, "ddg": ddg, "scorefile": str(scorefile)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rosetta = args.rosetta_bin
    if not rosetta.exists():
        raise SystemExit(f"Rosetta binary not found: {rosetta}")

    mutfiles = sorted(args.mutfiles_dir.glob("*.mutfile"))
    if not mutfiles:
        raise SystemExit(f"No mutfiles found in {args.mutfiles_dir}")

    results: List[Dict[str, object]] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for mutfile in mutfiles:
            results.append(_run_one(mutfile, args, rosetta))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            future_map = {ex.submit(_run_one, mf, args, rosetta): mf for mf in mutfiles}
            for fut in as_completed(future_map):
                results.append(fut.result())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ddg", "scorefile"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {args.out_csv} with {len(results)} rows")


if __name__ == "__main__":
    main()
