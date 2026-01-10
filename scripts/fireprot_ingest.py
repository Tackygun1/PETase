"""
Fetch FireProtDB sequences via API and produce sequence/label CSVs for surrogate training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.preprocess_sequences import clean_sequences

DEFAULT_BASE = "https://loschmidt.chemi.muni.cz/fireprotdb/api"


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    for col in df.columns:
        if col.upper() == name.upper():
            return col
    return None


def _infer_seq_for_model(df: pd.DataFrame) -> pd.Series:
    target_col = _find_column(df, "TARGET_SEQUENCE_ID")
    seq_col = _find_column(df, "SEQUENCE_ID")
    if target_col:
        base = df[target_col].replace(r"^\s*$", pd.NA, regex=True)
        if seq_col:
            fallback = df[seq_col].replace(r"^\s*$", pd.NA, regex=True)
            base = base.fillna(fallback)
        return base
    if seq_col:
        return df[seq_col].replace(r"^\s*$", pd.NA, regex=True)
    raise SystemExit("Missing SEQUENCE_ID/TARGET_SEQUENCE_ID columns in input CSV.")


def fetch_sequence(
    seq_id: int,
    session: requests.Session,
    base: str,
    retries: int = 5,
    sleep: float = 0.25,
) -> str:
    url = f"{base}/sequences/{seq_id}/sequence"
    last_error = "unknown"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return json.loads(resp.text)
            last_error = f"status={resp.status_code} body={resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(sleep * (2**attempt))
    raise RuntimeError(f"Failed to fetch sequence for id={seq_id}: {last_error}")


def _apply_label(df: pd.DataFrame, label_col: str, ddg_flip: bool) -> pd.DataFrame:
    col = _find_column(df, label_col)
    if not col:
        raise SystemExit(f"Label column {label_col} not found in input.")
    out = df.copy()
    out["stability"] = pd.to_numeric(out[col], errors="coerce")
    if ddg_flip and col.upper() == "DDG":
        out["stability"] = -out["stability"]
    return out.dropna(subset=["stability"])


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest FireProtDB CSV into sequence/label tables.")
    p.add_argument("input_csv", type=Path, help="FireProtDB CSV export.")
    p.add_argument(
        "--label-col",
        default="DTM",
        help="Label column to train on (e.g., DTM, DDG, TM).",
    )
    p.add_argument(
        "--out-seqs",
        type=Path,
        default=Path("data/processed/sequence_db.csv"),
        help="Output sequence DB CSV (id,sequence).",
    )
    p.add_argument(
        "--out-labels",
        type=Path,
        default=Path("data/processed/labels.csv"),
        help="Output labels CSV (id,stability).",
    )
    p.add_argument(
        "--id-source",
        choices=("sequence", "mutant", "experiment"),
        default="sequence",
        help="Identifier to use for training rows.",
    )
    p.add_argument(
        "--id-prefix",
        default="",
        help="Optional prefix to avoid ID collisions (e.g., fireprot_).",
    )
    p.add_argument(
        "--keep-indels",
        action="store_true",
        help="Keep rows with insertions/deletions (default drops them).",
    )
    p.add_argument(
        "--no-ddg-flip",
        action="store_true",
        help="Do not flip DDG sign (default flips so higher=better).",
    )
    p.add_argument(
        "--agg",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation for multiple rows per id.",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows.")
    p.add_argument("--max-seqs", type=int, default=None, help="Optional cap on sequences fetched.")
    p.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Filter to a specific SEQUENCE_LENGTH before fetching sequences.",
    )
    p.add_argument("--api-base", default=DEFAULT_BASE, help="FireProtDB API base URL.")
    p.add_argument("--retries", type=int, default=5, help="HTTP retries per sequence.")
    p.add_argument("--sleep", type=float, default=0.25, help="Initial backoff sleep (seconds).")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    df["SEQ_FOR_MODEL"] = _infer_seq_for_model(df)
    df["SEQ_FOR_MODEL"] = df["SEQ_FOR_MODEL"].replace(r"^\s*$", pd.NA, regex=True)
    df["SEQ_FOR_MODEL"] = pd.to_numeric(df["SEQ_FOR_MODEL"], errors="coerce")
    df = df.dropna(subset=["SEQ_FOR_MODEL"])
    df["SEQ_FOR_MODEL"] = df["SEQ_FOR_MODEL"].astype(int)

    if args.sequence_length is not None:
        len_col = _find_column(df, "SEQUENCE_LENGTH")
        if not len_col:
            raise SystemExit("SEQUENCE_LENGTH column not found; cannot filter by length.")
        lengths = pd.to_numeric(df[len_col], errors="coerce")
        df = df[lengths == args.sequence_length]

    df = _apply_label(df, args.label_col, ddg_flip=not args.no_ddg_flip)

    if not args.keep_indels:
        for col_name in ("INSERTION", "DELETION"):
            col = _find_column(df, col_name)
            if col:
                df = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]

    if args.max_rows:
        df = df.head(args.max_rows)

    if args.id_source == "sequence":
        id_raw = df["SEQ_FOR_MODEL"].astype(str)
    elif args.id_source == "mutant":
        col = _find_column(df, "MUTANT_ID")
        if not col:
            raise SystemExit("MUTANT_ID column not found.")
        id_raw = df[col].astype(str)
    else:
        col = _find_column(df, "EXPERIMENT_ID")
        if not col:
            raise SystemExit("EXPERIMENT_ID column not found.")
        id_raw = df[col].astype(str)

    df["id"] = args.id_prefix + id_raw

    seq_ids = sorted({int(x) for x in df["SEQ_FOR_MODEL"].unique()})
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    session = requests.Session()
    session.headers.update({"User-Agent": "fireprot-seq-fetch/1.0"})
    id_to_seq: dict[int, str] = {}
    for idx, sid in enumerate(seq_ids, 1):
        if idx % 100 == 0 or idx == len(seq_ids):
            print(f"Fetched {idx}/{len(seq_ids)} sequences")
        id_to_seq[sid] = fetch_sequence(
            sid, session=session, base=args.api_base, retries=args.retries, sleep=args.sleep
        )

    df["sequence"] = df["SEQ_FOR_MODEL"].map(id_to_seq)
    df = df.dropna(subset=["sequence"])

    seq_db = df[["id", "sequence"]].drop_duplicates("id")
    seq_db = clean_sequences(seq_db, seq_col="sequence")
    seq_db = seq_db[seq_db["sequence"].astype(str).str.len() > 0]

    if args.agg == "mean":
        labels = df.groupby("id", as_index=False)["stability"].mean()
    else:
        labels = df.groupby("id", as_index=False)["stability"].median()

    labels = labels[labels["id"].isin(seq_db["id"])]

    args.out_seqs.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    seq_db.to_csv(args.out_seqs, index=False)
    labels.to_csv(args.out_labels, index=False)

    print(f"Wrote {args.out_seqs} with {len(seq_db)} sequences")
    print(f"Wrote {args.out_labels} with {len(labels)} labels")


if __name__ == "__main__":
    main()
