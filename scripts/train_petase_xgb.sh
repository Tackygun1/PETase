#!/usr/bin/env bash
set -euo pipefail

# Workflow for PETase-only XGBoost training with ESM embeddings.

SEQ_PATH="${SEQ_PATH:-src/data/petase_sequences.csv}"
LABEL_PATH="${LABEL_PATH:-src/data/petase_tm_labels.csv}"
EMB_PATH="${EMB_PATH:-data/processed/petase_esm_embeddings.npz}"

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

if [ ! -f "$EMB_PATH" ]; then
  PYTHONPATH=. python3 -m src.utils.esm_embed \
    "$SEQ_PATH" \
    -o "$EMB_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH"
else
  echo "Embeddings already exist: $EMB_PATH"
fi

PYTHONPATH=. python3 -m src.models.train_xgb \
  --embeddings "$EMB_PATH" \
  --labels "$LABEL_PATH" \
  --target-col TM_C \
  --model-out models/petase_xgb_tm.json \
  --metrics-out models/petase_xgb_tm_metrics.json \
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

PYTHONPATH=. python3 -m src.models.train_xgb \
  --embeddings "$EMB_PATH" \
  --labels "$LABEL_PATH" \
  --target-col DTM_C \
  --model-out models/petase_xgb_dtm.json \
  --metrics-out models/petase_xgb_dtm_metrics.json \
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
