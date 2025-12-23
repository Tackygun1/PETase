"""
Aggregate per-round summaries into a single report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Summarize round results CSVs into one table.")
    p.add_argument("inputs", nargs="+", type=Path, help="Round CSVs (e.g., round_*_selected.csv).")
    p.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = p.parse_args()

    frames = []
    for inp in args.inputs:
        df = pd.read_csv(inp)
        df["source"] = inp.stem
        frames.append(df)
    if not frames:
        return
    out_df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
