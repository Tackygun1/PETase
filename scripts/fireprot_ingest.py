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


def _infer_seq_for_model(df: pd.DataFrame) -> pd.DataFrame:
    target_col = _find_column(df, "TARGET_SEQUENCE_ID")
    seq_col = _find_column(df, "SEQUENCE_ID")
    uniprot_col = _find_column(df, "UNIPROTKB")
    base = pd.Series(pd.NA, index=df.index)
    source = pd.Series(pd.NA, index=df.index)
    if target_col:
        base = df[target_col].replace(r"^\s*$", pd.NA, regex=True)
        source = source.mask(base.notna(), "fireprot")
    if seq_col:
        fallback = df[seq_col].replace(r"^\s*$", pd.NA, regex=True)
        base = base.fillna(fallback)
        source = source.mask(fallback.notna() & source.isna(), "fireprot")
    if uniprot_col:
        uniprot = df[uniprot_col].replace(r"^\s*$", pd.NA, regex=True)
        base = base.fillna(uniprot)
        source = source.mask(uniprot.notna() & source.isna(), "uniprot")
    if base.isna().all():
        raise SystemExit(
            "Missing SEQUENCE_ID/TARGET_SEQUENCE_ID/UNIPROTKB columns in input CSV."
        )
    return pd.DataFrame({"SEQ_FOR_MODEL": base, "SEQ_SOURCE": source})


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


def fetch_uniprot_sequence(
    uniprot_id: str,
    session: requests.Session,
    base: str,
    retries: int = 5,
    sleep: float = 0.25,
) -> str:
    url = f"{base}/{uniprot_id}.fasta"
    last_error = "unknown"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
                return "".join(line for line in lines if not line.startswith(">"))
            last_error = f"status={resp.status_code} body={resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(sleep * (2**attempt))
    raise RuntimeError(f"Failed to fetch UniProt sequence for id={uniprot_id}: {last_error}")


def _normalize_uniprot_id(raw: str) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for ch in [",", ";", "|"]:
        text = text.replace(ch, " ")
    parts = [p.strip() for p in text.split() if p.strip()]
    if not parts:
        return None
    return parts[0]


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
    p.add_argument(
        "--uniprot-base",
        default="https://rest.uniprot.org/uniprotkb",
        help="UniProt REST base URL for FASTA fetches.",
    )
    p.add_argument("--retries", type=int, default=5, help="HTTP retries per sequence.")
    p.add_argument("--sleep", type=float, default=0.25, help="Initial backoff sleep (seconds).")
    p.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip sequence fetch failures instead of aborting.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    seq_info = _infer_seq_for_model(df)
    df["SEQ_FOR_MODEL"] = seq_info["SEQ_FOR_MODEL"]
    df["SEQ_SOURCE"] = seq_info["SEQ_SOURCE"]
    df["SEQ_FOR_MODEL"] = df["SEQ_FOR_MODEL"].replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["SEQ_FOR_MODEL", "SEQ_SOURCE"])
    df["SEQ_FOR_MODEL"] = df["SEQ_FOR_MODEL"].astype(str)
    uniprot_mask = df["SEQ_SOURCE"] == "uniprot"
    if uniprot_mask.any():
        df.loc[uniprot_mask, "SEQ_FOR_MODEL"] = df.loc[uniprot_mask, "SEQ_FOR_MODEL"].map(
            _normalize_uniprot_id
        )
        df = df.dropna(subset=["SEQ_FOR_MODEL"])

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

    session = requests.Session()
    session.headers.update({"User-Agent": "fireprot-seq-fetch/1.0"})
    fireprot_raw = df.loc[df["SEQ_SOURCE"] == "fireprot", "SEQ_FOR_MODEL"]
    fireprot_ids = pd.to_numeric(fireprot_raw, errors="coerce").dropna().astype(int).unique()
    fireprot_ids = sorted(set(fireprot_ids))
    uniprot_ids = sorted(
        {x for x in df.loc[df["SEQ_SOURCE"] == "uniprot", "SEQ_FOR_MODEL"].unique()}
    )
    if args.max_seqs:
        fireprot_ids = fireprot_ids[: args.max_seqs]
        uniprot_ids = uniprot_ids[: args.max_seqs]

    id_to_seq: dict[str, str] = {}
    total = len(fireprot_ids) + len(uniprot_ids)
    done = 0
    for sid in fireprot_ids:
        done += 1
        if done % 100 == 0 or done == total:
            print(f"Fetched {done}/{total} sequences")
        try:
            id_to_seq[str(sid)] = fetch_sequence(
                sid, session=session, base=args.api_base, retries=args.retries, sleep=args.sleep
            )
        except Exception as exc:
            if args.skip_failures:
                print(f"Warning: failed to fetch FireProt id={sid}: {exc}")
                continue
            raise
    for uid in uniprot_ids:
        done += 1
        if done % 100 == 0 or done == total:
            print(f"Fetched {done}/{total} sequences")
        try:
            id_to_seq[uid] = fetch_uniprot_sequence(
                uid, session=session, base=args.uniprot_base, retries=args.retries, sleep=args.sleep
            )
        except Exception as exc:
            if args.skip_failures:
                print(f"Warning: failed to fetch UniProt id={uid}: {exc}")
                continue
            raise

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
