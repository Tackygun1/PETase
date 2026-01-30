#!/usr/bin/env python3
"""
Export FireProtDB experimental records with context metadata and context keys.
Step 1/2/3 from the bias-mitigation workflow.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

SINGLE_SUB_RE = re.compile(r"^[A-Z][0-9]+[A-Z]$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FireProtDB records with context keys.")
    p.add_argument("input_csv", type=Path, help="FireProtDB CSV export.")
    p.add_argument(
        "--out-records",
        type=Path,
        default=Path("data/processed/fireprot_context_records.csv"),
        help="Output cleaned records CSV.",
    )
    p.add_argument(
        "--out-agg",
        type=Path,
        default=Path("data/processed/fireprot_context_agg.csv"),
        help="Output aggregated (median) CSV by mutant+context.",
    )
    p.add_argument(
        "--ph-bin",
        type=float,
        default=0.5,
        help="Bin size for pH (e.g. 0.5 or 1.0).",
    )
    p.add_argument(
        "--label-prefer",
        choices=("DTM", "TM"),
        default="DTM",
        help="Preferred label column when both exist.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Optional chunksize for CSV loading.",
    )
    return p.parse_args()


def _ph_bin(val: float | None, bin_size: float) -> str:
    if val is None or math.isnan(val):
        return "NA"
    return f"{round(val / bin_size) * bin_size:.2f}"


def _normalize_str(s: str | None) -> str:
    if s is None:
        return "NA"
    out = str(s).strip()
    return out if out else "NA"


def _label_from_row(row: pd.Series, prefer: str) -> tuple[float | None, str | None]:
    dtm = pd.to_numeric(row.get("DTM"), errors="coerce")
    tm = pd.to_numeric(row.get("TM"), errors="coerce")
    if prefer == "DTM":
        if not pd.isna(dtm):
            return float(dtm), "DTM"
        if not pd.isna(tm):
            return float(tm), "TM"
    else:
        if not pd.isna(tm):
            return float(tm), "TM"
        if not pd.isna(dtm):
            return float(dtm), "DTM"
    return None, None


def _is_single_sub(sub: str) -> bool:
    return bool(SINGLE_SUB_RE.match(sub))


def _clean_df(df: pd.DataFrame, prefer: str, ph_bin: float) -> pd.DataFrame:
    # Filter to single-point substitutions only and no indels.
    df = df.copy()
    df["SUBSTITUTION"] = df["SUBSTITUTION"].astype(str).str.strip()
    df = df[df["SUBSTITUTION"].apply(_is_single_sub)]
    for col in ("INSERTION", "DELETION"):
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            df = df[mask]

    # Label selection (DTM preferred, TM fallback)
    labels = []
    label_types = []
    for _, row in df.iterrows():
        label, ltype = _label_from_row(row, prefer)
        labels.append(label)
        label_types.append(ltype)
    df["label"] = labels
    df["label_type"] = label_types
    df = df.dropna(subset=["label", "label_type"])

    # Require UniProt ID for sequence reconstruction
    df["UNIPROTKB"] = df["UNIPROTKB"].astype(str).str.strip()
    df = df[df["UNIPROTKB"].notna() & (df["UNIPROTKB"] != "") & (df["UNIPROTKB"] != "nan")]

    # Normalize context fields
    df["publication"] = df["PUBLICATION_PMID"].fillna("").astype(str).str.strip()
    df["publication"] = df["publication"].mask(df["publication"] == "", df["PUBLICATION_DOI"])
    df["publication"] = df["publication"].apply(_normalize_str)
    df["measure"] = df["MEASURE"].apply(_normalize_str)
    df["method"] = df["METHOD"].apply(_normalize_str)

    df["PH"] = pd.to_numeric(df["PH"], errors="coerce")
    df["ph_bin"] = df["PH"].apply(lambda v: _ph_bin(v, ph_bin))

    # Optional context fields
    df["buffer"] = df["BUFFER"].apply(_normalize_str)
    df["buffer_conc"] = df["BUFFER_CONC"].apply(_normalize_str)
    df["exp_temperature"] = df["EXP_TEMPERATURE"].apply(_normalize_str)
    df["ion"] = df["ION"].apply(_normalize_str)
    df["ion_conc"] = df["ION_CONC"].apply(_normalize_str)

    df["context_key"] = (
        df["publication"]
        + "|"
        + df["measure"]
        + "|"
        + df["method"]
        + "|"
        + df["ph_bin"]
    )

    return df


def _select_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "EXPERIMENT_ID",
        "MUTANT_ID",
        "UNIPROTKB",
        "SUBSTITUTION",
        "label",
        "label_type",
        "publication",
        "measure",
        "method",
        "PH",
        "ph_bin",
        "buffer",
        "buffer_conc",
        "exp_temperature",
        "ion",
        "ion_conc",
        "context_key",
    ]
    return df[keep]


def main() -> None:
    args = parse_args()
    usecols = [
        "EXPERIMENT_ID",
        "MUTANT_ID",
        "UNIPROTKB",
        "SUBSTITUTION",
        "INSERTION",
        "DELETION",
        "DTM",
        "TM",
        "MEASURE",
        "METHOD",
        "PH",
        "BUFFER",
        "BUFFER_CONC",
        "EXP_TEMPERATURE",
        "ION",
        "ION_CONC",
        "PUBLICATION_PMID",
        "PUBLICATION_DOI",
    ]

    chunks: List[pd.DataFrame] = []
    if args.chunksize:
        for chunk in pd.read_csv(args.input_csv, usecols=usecols, chunksize=args.chunksize):
            clean = _clean_df(chunk, args.label_prefer, args.ph_bin)
            chunks.append(_select_cols(clean))
    else:
        df = pd.read_csv(args.input_csv, usecols=usecols, low_memory=False)
        clean = _clean_df(df, args.label_prefer, args.ph_bin)
        chunks.append(_select_cols(clean))

    if not chunks:
        raise SystemExit("No records after filtering.")
    records = pd.concat(chunks, ignore_index=True)

    args.out_records.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.out_records, index=False)

    # Aggregate duplicates by mutant + context
    agg = (
        records.groupby(["UNIPROTKB", "SUBSTITUTION", "context_key"], as_index=False)
        .agg(
            label=("label", "median"),
            label_type=("label_type", "first"),
            publication=("publication", "first"),
            measure=("measure", "first"),
            method=("method", "first"),
            PH=("PH", "median"),
            ph_bin=("ph_bin", "first"),
            buffer=("buffer", "first"),
            buffer_conc=("buffer_conc", "first"),
            exp_temperature=("exp_temperature", "first"),
            ion=("ion", "first"),
            ion_conc=("ion_conc", "first"),
            n_records=("label", "count"),
        )
    )
    args.out_agg.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_agg, index=False)

    print(f"Wrote {args.out_records} with {len(records)} records")
    print(f"Wrote {args.out_agg} with {len(agg)} aggregated rows")


if __name__ == "__main__":
    main()
