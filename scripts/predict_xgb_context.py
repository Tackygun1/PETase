#!/usr/bin/env python3
"""
Predict residualized (and optionally baseline-adjusted) stability using a context XGBoost model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with a context-centered XGBoost model.")
    p.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Embeddings NPZ keyed by id.",
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="XGBoost model JSON.",
    )
    p.add_argument(
        "--baselines",
        type=Path,
        default=None,
        help="Baseline JSON from train_xgb_context.py (optional).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset CSV with id/context_key/publication for per-row baselines.",
    )
    p.add_argument(
        "--target-context-key",
        default=None,
        help="Override context key for all predictions.",
    )
    p.add_argument(
        "--target-publication",
        default=None,
        help="Override publication for all predictions.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/surrogate_af2/context_predictions.csv"),
        help="Output predictions CSV.",
    )
    p.add_argument(
        "--plot-hist",
        type=Path,
        default=None,
        help="Optional path to save a histogram PNG.",
    )
    return p.parse_args()


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_baselines(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _get_stats(stats: Dict[str, Dict[str, object]], key: str) -> Tuple[float, int] | None:
    if key not in stats:
        return None
    entry = stats[key]
    try:
        median = float(entry.get("median"))
        count = int(entry.get("count"))
    except Exception:
        return None
    return median, count


def build_baseline_fn(cfg: Dict[str, object]):
    global_med = float(cfg["global_median"])
    pub_stats = cfg.get("pub_stats", {})
    ctx_stats = cfg.get("ctx_stats", {})
    min_context_n = int(cfg.get("min_context_n", 20))
    min_pub_n = int(cfg.get("min_pub_n", 10))
    shrink_lambda = float(cfg.get("shrink_lambda", 20.0))

    def pub_baseline(pub: str) -> float:
        stats = _get_stats(pub_stats, pub)
        if not stats:
            return global_med
        med, n = stats
        if n < min_pub_n:
            return global_med
        return (n * med + shrink_lambda * global_med) / (n + shrink_lambda)

    def ctx_baseline(ctx: str, pub: str) -> float:
        stats = _get_stats(ctx_stats, ctx)
        b_pub = pub_baseline(pub)
        if not stats:
            return b_pub
        med, n = stats
        if n < min_context_n:
            return b_pub
        return (n * med + shrink_lambda * b_pub) / (n + shrink_lambda)

    return ctx_baseline


def main() -> None:
    args = parse_args()
    embeddings = load_embeddings(args.embeddings)
    if not embeddings:
        raise SystemExit("No embeddings found.")

    if args.dataset:
        df = pd.read_csv(args.dataset)
        if "id" not in df.columns:
            raise SystemExit("Dataset must include an 'id' column.")
        df = df[df["id"].isin(embeddings.keys())].reset_index(drop=True)
        if df.empty:
            raise SystemExit("No overlapping IDs between dataset and embeddings.")
    else:
        df = pd.DataFrame({"id": list(embeddings.keys())})

    X = np.vstack([embeddings[rid] for rid in df["id"].tolist()])

    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(args.model)
    pred = model.predict(X)

    out = df.copy()
    out["pred_residual"] = pred

    if args.baselines:
        cfg = load_baselines(args.baselines)
        ctx_fn = build_baseline_fn(cfg)
        ctx_keys = []
        pubs = []
        baselines = []
        for _, row in out.iterrows():
            ctx = args.target_context_key or str(row.get("context_key", "NA"))
            pub = args.target_publication or str(row.get("publication", "NA"))
            ctx_keys.append(ctx)
            pubs.append(pub)
            baselines.append(ctx_fn(ctx, pub))
        out["context_key"] = ctx_keys
        out["publication"] = pubs
        out["baseline"] = baselines
        out["pred_abs"] = out["pred_residual"] + out["baseline"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")

    if args.plot_hist:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping histogram plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.hist(out["pred_residual"], bins=40, alpha=0.7, label="residual")
        if "pred_abs" in out.columns:
            plt.hist(out["pred_abs"], bins=40, alpha=0.7, label="absolute")
        plt.xlabel("Predicted value")
        plt.ylabel("Count")
        plt.legend()
        args.plot_hist.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot_hist, dpi=180)
        plt.close()
        print(f"Wrote histogram to {args.plot_hist}")


if __name__ == "__main__":
    main()
