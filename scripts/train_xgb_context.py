#!/usr/bin/env python3
"""
Train a context-centered XGBoost surrogate on FireProtDB with GroupKFold by publication.
Implements steps 4–7 of the bias-mitigation workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Context-centered XGBoost training.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/fireprot_context_dataset.csv"),
        help="Dataset CSV with id, label, context_key, publication.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/fireprot_context_embeddings.npz"),
        help="Embeddings NPZ keyed by id.",
    )
    p.add_argument(
        "--label-col",
        default="label",
        help="Label column name in dataset (default: label).",
    )
    p.add_argument(
        "--group-col",
        default="publication",
        help="Group column for GroupKFold (default: publication).",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--min-context-n", type=int, default=20)
    p.add_argument("--min-pub-n", type=int, default=10)
    p.add_argument("--shrink-lambda", type=float, default=20.0)
    p.add_argument(
        "--quantile-alpha",
        type=float,
        default=0.2,
        help="Quantile alpha for reg:quantileerror.",
    )
    p.add_argument(
        "--objective",
        default="reg:quantileerror",
        help="XGBoost objective (default: reg:quantileerror).",
    )
    p.add_argument("--tree-method", default="hist")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-child-weight", type=float, default=5)
    p.add_argument("--subsample", type=float, default=0.7)
    p.add_argument("--colsample-bytree", type=float, default=0.7)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/fireprot_context_xgb_dtm_q20.json"),
        help="Output model JSON.",
    )
    p.add_argument(
        "--baseline-out",
        type=Path,
        default=Path("models/fireprot_context_baselines.json"),
        help="Output JSON with baselines and config.",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("models/fireprot_context_metrics.csv"),
        help="Output metrics CSV.",
    )
    return p.parse_args()


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def compute_baselines(
    df: pd.DataFrame,
    label_col: str,
    min_context_n: int,
    min_pub_n: int,
    shrink_lambda: float,
) -> Tuple[float, Dict[str, Tuple[float, int]], Dict[str, Tuple[float, int]]]:
    global_med = float(df[label_col].median())
    pub_stats = (
        df.groupby("publication")[label_col]
        .agg(["median", "count"])
        .to_dict(orient="index")
    )
    ctx_stats = (
        df.groupby("context_key")[label_col]
        .agg(["median", "count"])
        .to_dict(orient="index")
    )

    def pub_baseline(pub: str) -> float:
        stats = pub_stats.get(pub)
        if not stats:
            return global_med
        n = int(stats["count"])
        b = float(stats["median"])
        if n < min_pub_n:
            return global_med
        return (n * b + shrink_lambda * global_med) / (n + shrink_lambda)

    def ctx_baseline(ctx: str, pub: str) -> float:
        stats = ctx_stats.get(ctx)
        b_pub = pub_baseline(pub)
        if not stats:
            return b_pub
        n = int(stats["count"])
        b = float(stats["median"])
        if n < min_context_n:
            return b_pub
        return (n * b + shrink_lambda * b_pub) / (n + shrink_lambda)

    return global_med, pub_stats, ctx_stats, pub_baseline, ctx_baseline


def residualize(
    df: pd.DataFrame,
    label_col: str,
    pub_baseline_fn,
    ctx_baseline_fn,
) -> np.ndarray:
    residuals = []
    for _, row in df.iterrows():
        pub = str(row["publication"])
        ctx = str(row["context_key"])
        base = ctx_baseline_fn(ctx, pub)
        residuals.append(float(row[label_col]) - base)
    return np.array(residuals, dtype=float)


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = []
    for token in version.split("."):
        num = ""
        for ch in token:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        parts.append(int(num))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def train_xgb(args: argparse.Namespace, X: np.ndarray, y: np.ndarray):
    import xgboost as xgb

    objective = args.objective
    if objective == "reg:quantileerror":
        version = _parse_version(getattr(xgb, "__version__", "0.0.0"))
        if version < (2, 0, 0):
            raise SystemExit(
                "XGBoost >= 2.0.0 is required for reg:quantileerror. "
                "Upgrade xgboost or pass --objective reg:squarederror."
            )
    params = dict(
        objective=objective,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        learning_rate=args.learning_rate,
        tree_method=args.tree_method,
        random_state=args.seed,
    )
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = args.quantile_alpha
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column {args.label_col} not found in dataset.")
    if args.group_col not in df.columns:
        raise SystemExit(f"Group column {args.group_col} not found in dataset.")

    embeddings = load_embeddings(args.embeddings)
    df = df[df["id"].isin(embeddings.keys())].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No overlapping IDs between dataset and embeddings.")

    X = np.vstack([embeddings[rid] for rid in df["id"].tolist()])
    groups = df[args.group_col].astype(str).values
    n_groups = len(np.unique(groups))
    if n_groups < args.cv_folds:
        if n_groups < 2:
            raise SystemExit("Need at least 2 unique groups for GroupKFold.")
        print(f"Warning: only {n_groups} groups, reducing cv-folds to {n_groups}.")
        args.cv_folds = n_groups

    gkf = GroupKFold(n_splits=args.cv_folds)
    fold_rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=groups), start=1):
        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()

        global_med, pub_stats, ctx_stats, pub_fn, ctx_fn = compute_baselines(
            train_df,
            label_col=args.label_col,
            min_context_n=args.min_context_n,
            min_pub_n=args.min_pub_n,
            shrink_lambda=args.shrink_lambda,
        )
        y_tr = residualize(train_df, args.label_col, pub_fn, ctx_fn)
        y_te = residualize(test_df, args.label_col, pub_fn, ctx_fn)

        model = train_xgb(args, X[tr_idx], y_tr)
        pred = model.predict(X[te_idx])
        fold_rows.append(
            {
                "fold": fold,
                "rmse": rmse(y_te, pred),
                "mae": mae(y_te, pred),
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
            }
        )

    # Train final model on full dataset with baselines from full data
    global_med, pub_stats, ctx_stats, pub_fn, ctx_fn = compute_baselines(
        df,
        label_col=args.label_col,
        min_context_n=args.min_context_n,
        min_pub_n=args.min_pub_n,
        shrink_lambda=args.shrink_lambda,
    )
    y_full = residualize(df, args.label_col, pub_fn, ctx_fn)
    final_model = train_xgb(args, X, y_full)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(args.model_out)

    # Save baselines for later inference
    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.baseline_out, "w") as f:
        json.dump(
            {
                "global_median": global_med,
                "pub_stats": pub_stats,
                "ctx_stats": ctx_stats,
                "min_context_n": args.min_context_n,
                "min_pub_n": args.min_pub_n,
                "shrink_lambda": args.shrink_lambda,
                "label_col": args.label_col,
            },
            f,
        )

    # Write metrics
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    print(f"Saved model to {args.model_out}")
    print(f"Wrote baselines to {args.baseline_out}")
    print(f"Wrote CV metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
