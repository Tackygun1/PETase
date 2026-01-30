#!/usr/bin/env python3
"""
Build mutant sequences from FireProt context table using UniProt WT sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
import re


SUB_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mutant sequences from context table.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/fireprot_context_agg.csv"),
        help="Aggregated context CSV.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/fireprot_context_dataset.csv"),
        help="Output dataset CSV with sequences and labels.",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("data/processed/uniprot_cache.json"),
        help="JSON cache for UniProt sequences.",
    )
    p.add_argument(
        "--uniprot-base",
        default="https://rest.uniprot.org/uniprotkb",
        help="UniProt REST base URL for FASTA fetches.",
    )
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--skip-mismatch", action="store_true", help="Skip if WT AA mismatch.")
    return p.parse_args()


def _load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def fetch_uniprot_sequence(
    uniprot_id: str,
    base: str,
    retries: int,
    sleep: float,
) -> str:
    url = f"{base}/{uniprot_id}.fasta"
    last_error = "unknown"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
                return "".join(line for line in lines if not line.startswith(">"))
            last_error = f"status={resp.status_code} body={resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(sleep * (2**attempt))
    raise RuntimeError(f"Failed to fetch UniProt sequence for id={uniprot_id}: {last_error}")


def apply_substitution(wt: str, sub: str, skip_mismatch: bool) -> str | None:
    m = SUB_RE.match(sub)
    if not m:
        return None
    ref, pos_str, alt = m.groups()
    pos = int(pos_str)
    if pos < 1 or pos > len(wt):
        return None
    if wt[pos - 1] != ref:
        if skip_mismatch:
            return None
        raise ValueError(f"WT mismatch at {sub}: expected {ref}, found {wt[pos-1]}")
    return wt[: pos - 1] + alt + wt[pos:]


def context_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    cache = _load_cache(args.cache)

    rows = []
    fetch_fail = 0
    mismatch = 0
    invalid = 0
    for _, row in df.iterrows():
        uniprot = str(row["UNIPROTKB"]).strip()
        sub = str(row["SUBSTITUTION"]).strip()
        ctx = str(row["context_key"]).strip()
        if not uniprot or not sub or not ctx:
            continue
        wt = cache.get(uniprot)
        if wt is None:
            try:
                wt = fetch_uniprot_sequence(uniprot, args.uniprot_base, args.retries, args.sleep)
            except Exception:
                fetch_fail += 1
                continue
            cache[uniprot] = wt
        try:
            mut = apply_substitution(wt, sub, args.skip_mismatch)
        except ValueError:
            if args.skip_mismatch:
                mismatch += 1
                continue
            raise
        if not mut:
            invalid += 1
            continue
        cid = f"fpctx_{uniprot}_{sub}_{context_hash(ctx)}"
        rows.append(
            {
                "id": cid,
                "sequence": mut,
                "label": row["label"],
                "label_type": row["label_type"],
                "context_key": ctx,
                "publication": row["publication"],
                "measure": row["measure"],
                "method": row["method"],
                "ph_bin": row["ph_bin"],
                "uniprot": uniprot,
                "substitution": sub,
                "n_records": row["n_records"],
            }
        )

    _save_cache(args.cache, cache)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")
    if fetch_fail or mismatch or invalid:
        print(
            f"Skipped rows - fetch_fail={fetch_fail}, mismatch={mismatch}, invalid={invalid}"
        )


if __name__ == "__main__":
    main()
