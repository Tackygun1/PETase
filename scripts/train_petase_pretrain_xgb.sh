#!/usr/bin/env bash
set -euo pipefail

# PETase similarity pretrain workflow.

SEQ_PATH="${SEQ_PATH:-src/data/petase_sequences.csv}"
PETASE_LABELS="${PETASE_LABELS:-src/data/petase_tm_labels.csv}"
PETASE_LABEL_COL="${PETASE_LABEL_COL:-DTM_C}"

REF_EMB="${REF_EMB:-data/processed/esm_embeddings.npz}"
REF_LABELS="${REF_LABELS:-data/processed/labels.csv}"
REF_LABEL_COL="${REF_LABEL_COL:-stability}"

PETASE_EMB="${PETASE_EMB:-data/processed/petase_esm_embeddings.npz}"
PRETRAIN_EMB="${PRETRAIN_EMB:-data/processed/petase_pretrain_embeddings.npz}"
PRETRAIN_LABELS="${PRETRAIN_LABELS:-data/processed/petase_pretrain_labels.csv}"

TARGET_COL="${TARGET_COL:-DTM_C}"
TOP_K="${TOP_K:-200}"
MIN_SIM="${MIN_SIM:-}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1000}"

TREE_METHOD="${TREE_METHOD:-hist}"
N_ESTIMATORS="${N_ESTIMATORS:-300}"
MAX_DEPTH="${MAX_DEPTH:-2}"
MIN_CHILD_WEIGHT="${MIN_CHILD_WEIGHT:-5}"
SUBSAMPLE="${SUBSAMPLE:-0.7}"
COLSAMPLE="${COLSAMPLE:-0.7}"
REG_LAMBDA="${REG_LAMBDA:-10}"
REG_ALPHA="${REG_ALPHA:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
CV_FOLDS="${CV_FOLDS:-4}"

if [ ! -f "$PETASE_EMB" ]; then
  PYTHONPATH=. python3 -m src.utils.esm_embed \
    "$SEQ_PATH" \
    -o "$PETASE_EMB" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH"
else
  echo "PETase embeddings already exist: $PETASE_EMB"
fi

MIN_SIM_FLAG=()
if [ -n "$MIN_SIM" ]; then
  MIN_SIM_FLAG=(--min-sim "$MIN_SIM")
fi

PYTHONPATH=. python3 scripts/build_petase_pretrain_set.py \
  --ref-embeddings "$REF_EMB" \
  --ref-labels "$REF_LABELS" \
  --ref-label-col "$REF_LABEL_COL" \
  --petase-embeddings "$PETASE_EMB" \
  --petase-labels "$PETASE_LABELS" \
  --petase-label-col "$PETASE_LABEL_COL" \
  --target-col "$TARGET_COL" \
  --top-k "$TOP_K" \
  "${MIN_SIM_FLAG[@]}" \
  --out-embeddings "$PRETRAIN_EMB" \
  --out-labels "$PRETRAIN_LABELS"

PYTHONPATH=. python3 -m src.models.train_xgb \
  --embeddings "$PRETRAIN_EMB" \
  --labels "$PRETRAIN_LABELS" \
  --target-col "$TARGET_COL" \
  --model-out models/petase_pretrain_xgb_dtm.json \
  --metrics-out models/petase_pretrain_xgb_dtm_metrics.json \
  --tree-method "$TREE_METHOD" \
  --n-estimators "$N_ESTIMATORS" \
  --max-depth "$MAX_DEPTH" \
  --min-child-weight "$MIN_CHILD_WEIGHT" \
  --subsample "$SUBSAMPLE" \
  --colsample-bytree "$COLSAMPLE" \
  --reg-lambda "$REG_LAMBDA" \
  --reg-alpha "$REG_ALPHA" \
  --learning-rate "$LEARNING_RATE" \
  --cv-folds "$CV_FOLDS"
