"""
Train an XGBoost regressor on precomputed embeddings + labels.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from ..utils.io import load_embeddings, load_labels_csv
from .utils import align_X_y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _summarize_metrics(metrics: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not metrics:
        return summary
    keys = metrics[0].keys()
    for key in keys:
        vals = np.array([m[key] for m in metrics], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1 if len(vals) > 1 else 0)),
        }
    return summary


def _filter_fit_kwargs(model, fit_kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        params = set(inspect.signature(model.fit).parameters.keys())
    except (TypeError, ValueError):
        return fit_kwargs
    return {k: v for k, v in fit_kwargs.items() if k in params}


def _fit_kwargs(
    model,
    args: argparse.Namespace,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
) -> Dict[str, object]:
    if X_val is None or y_val is None or not args.early_stopping:
        return {}
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "early_stopping_rounds": args.early_stopping,
        "verbose": False,
    }
    filtered = _filter_fit_kwargs(model, fit_kwargs)
    if "early_stopping_rounds" not in filtered:
        print("Warning: early stopping not supported by this XGBoost version; skipping.")
    if "eval_set" not in filtered and "early_stopping_rounds" in filtered:
        print("Warning: eval_set unsupported; early stopping disabled.")
        filtered.pop("early_stopping_rounds", None)
    return filtered


def _build_model(args: argparse.Namespace):
    import xgboost as xgb

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )


def _maybe_set_eval_metric(model, metric: Optional[str]) -> None:
    if not metric:
        return
    try:
        model.set_params(eval_metric=metric)
    except TypeError:
        # Older XGBoost sklearn APIs may not accept eval_metric here.
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost regressor from embeddings/labels.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/esm_embeddings.npz"),
        help="Path to embeddings .npz (default: data/processed/esm_embeddings.npz).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/labels.csv"),
        help="Path to labels CSV (default: data/processed/labels.csv).",
    )
    parser.add_argument("--id-col", default="id", help="ID column name in labels CSV.")
    parser.add_argument(
        "--target-col",
        default="stability",
        help="Target column to predict (default: stability).",
    )
    parser.add_argument("--dropna", action="store_true", default=True, help="Drop NaN targets.")
    parser.add_argument("--no-dropna", dest="dropna", action="store_false")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction (0-1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle before split.")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >1, run K-fold CV (overrides test split for evaluation).",
    )
    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=42,
        help="Random seed for CV shuffling.",
    )
    parser.add_argument(
        "--cv-shuffle",
        action="store_true",
        default=True,
        help="Shuffle before CV split.",
    )
    parser.add_argument("--no-cv-shuffle", dest="cv_shuffle", action="store_false")

    # XGBoost params
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--tree-method", default="hist", help='XGBoost tree_method (e.g. "hist").')
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping rounds (requires a validation split or CV fold).",
    )
    parser.add_argument(
        "--eval-metric",
        default="rmse",
        help="Eval metric for early stopping (default: rmse).",
    )

    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/xgb_surrogate.json"),
        help="Path to save trained model (default: models/xgb_surrogate.json).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    return parser.parse_args()


def _maybe_write_metrics(metrics: Dict[str, object], out_path: Optional[Path]) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()

    embeddings = load_embeddings(args.embeddings)
    labels_df = load_labels_csv(args.labels, id_col=args.id_col, y_col=args.target_col)
    if args.dropna:
        labels_df = labels_df.dropna(subset=[args.target_col])
    if labels_df.empty:
        raise SystemExit("No labels available after filtering.")

    X, y, kept_ids = align_X_y(
        embeddings, labels_df, id_col=args.id_col, y_col=args.target_col
    )
    if y.ndim != 2 or y.shape[1] != 1:
        raise SystemExit("XGBoost training supports a single target column.")
    y = y.reshape(-1)

    cv_report = None
    if args.cv_folds and args.cv_folds > 1:
        cv = KFold(
            n_splits=args.cv_folds,
            shuffle=args.cv_shuffle,
            random_state=args.cv_random_state if args.cv_shuffle else None,
        )
        fold_metrics = []
        for train_idx, val_idx in cv.split(X):
            model = _build_model(args)
            _maybe_set_eval_metric(model, args.eval_metric)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model.fit(X_train, y_train, **_fit_kwargs(model, args, X_val, y_val))
            preds = model.predict(X_val)
            fold_metrics.append(_compute_metrics(y_val, preds))
        cv_report = {
            "folds": int(args.cv_folds),
            "summary": _summarize_metrics(fold_metrics),
            "per_fold": fold_metrics,
        }
        X_tr, y_tr = X, y
        X_te = y_te = None
    elif args.test_size and args.test_size > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=args.shuffle,
        )
    else:
        X_tr, y_tr = X, y
        X_te = y_te = None

    model = _build_model(args)
    _maybe_set_eval_metric(model, args.eval_metric)
    model.fit(X_tr, y_tr, **_fit_kwargs(model, args, X_te, y_te))

    metrics: Dict[str, object] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(0 if X_te is None else X_te.shape[0]),
        "train": _compute_metrics(y_tr, model.predict(X_tr)),
    }
    if cv_report is not None:
        metrics["cv"] = cv_report
    if X_te is not None:
        metrics["test"] = _compute_metrics(y_te, model.predict(X_te))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)

    print(f"Trained XGBoost on {metrics['n_train']} samples (features={metrics['n_features']}).")
    if cv_report is not None:
        cv_rmse = cv_report["summary"]["rmse"]
        print(f"CV RMSE: {cv_rmse['mean']:.4f} ± {cv_rmse['std']:.4f}")
    if X_te is not None:
        print(f"Holdout size: {metrics['n_test']}")
        print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    print(f"Saved model to {args.model_out}")

    _maybe_write_metrics(metrics, args.metrics_out)


if __name__ == "__main__":
    main()
