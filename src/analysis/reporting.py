"""
Simple reporting helpers for round summaries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_round_summary(selected, out_path: Path) -> None:
    """Persist selected candidates and key scores to CSV."""
    rows = []
    for rec in selected:
        rows.append(
            {
                "seq_id": rec.seq_id,
                "mutation_count": rec.mutation_count,
                "stability_score": rec.stability_score,
                "activity_score": rec.activity_score,
                "composite": rec.composite,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


__all__ = ["write_round_summary"]
