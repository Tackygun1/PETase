#!/usr/bin/env bash
set -euo pipefail

# End-to-end overnight pipeline:
# 1) Ensure WT FASTA + PETase embeddings exist
# 2) Merge reference embeddings + sequence DB with PETase
# 3) Generate diverse candidates (retrieval pool)
# 4) Embed candidates
# 5) Rank with XGBoost surrogate and stage AF2/ColabFold inputs

WT_ID="${WT_ID:-IsPETaseWT}"
WT_FASTA="${WT_FASTA:-data/processed/petase_wt.fasta}"
PETASE_SEQ_CSV="${PETASE_SEQ_CSV:-src/data/petase_sequences.csv}"
PETASE_EMB="${PETASE_EMB:-data/processed/petase_esm_embeddings.npz}"

REF_EMB="${REF_EMB:-}"
REF_SEQ_DB="${REF_SEQ_DB:-}"
MERGED_EMB="${MERGED_EMB:-data/processed/esm_embeddings_tm_plus_petase.npz}"
MERGED_SEQ_DB="${MERGED_SEQ_DB:-data/processed/sequence_db_petase_all.csv}"

PROPOSALS="${PROPOSALS:-5000}"
PROPOSALS_PER_NEIGHBOR="${PROPOSALS_PER_NEIGHBOR:-4}"
TRUST_MIN="${TRUST_MIN:-1}"
TRUST_MAX="${TRUST_MAX:-5}"
PROPOSAL_MODE="${PROPOSAL_MODE:-pool}"
SEED="${SEED:-42}"
EXTRA_PROTECTED="${EXTRA_PROTECTED:-}"

CAND_FASTA="${CAND_FASTA:-data/processed/candidates.fasta}"
CAND_EMB="${CAND_EMB:-data/processed/candidates_esm.npz}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1000}"
ESM_MODEL="${ESM_MODEL:-}"

MODEL_XGB="${MODEL_XGB:-models/petase_pretrain_xgb_dtm.json}"
TOP_K="${TOP_K:-20}"

OUT_DIR="${OUT_DIR:-results/surrogate_af2}"
RUN_AF2="${RUN_AF2:-1}"
AF2_BIN="${AF2_BIN:-}"

if [ -z "$REF_EMB" ]; then
  if [ -f data/processed/esm_embeddings_dtm.npz ]; then
    REF_EMB="data/processed/esm_embeddings_dtm.npz"
  else
    REF_EMB="data/processed/esm_embeddings.npz"
  fi
fi
if [ -z "$REF_SEQ_DB" ]; then
  if [ -f data/processed/sequence_db_dtm.csv ]; then
    REF_SEQ_DB="data/processed/sequence_db_dtm.csv"
  else
    REF_SEQ_DB="data/processed/sequence_db.csv"
  fi
fi

if [ ! -f "$WT_FASTA" ]; then
  python3 - <<PY
import csv, pathlib
wt_id = "${WT_ID}"
src = "${PETASE_SEQ_CSV}"
out = pathlib.Path("${WT_FASTA}")
seq = None
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("id","").strip() == wt_id:
            seq = row.get("sequence","").strip()
            break
if not seq:
    raise SystemExit(f"WT id {wt_id} not found in {src}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(f">{wt_id}\n{seq}\n")
print("Wrote", out)
PY
fi

if [ ! -f "$PETASE_EMB" ]; then
  ESM_FLAGS=()
  if [ -n "$ESM_MODEL" ]; then
    ESM_FLAGS=(--model "$ESM_MODEL")
  fi
  PYTHONPATH=. python3 -m src.utils.esm_embed \
    "$PETASE_SEQ_CSV" \
    -o "$PETASE_EMB" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    "${ESM_FLAGS[@]}"
fi

if [ ! -f "$MERGED_EMB" ]; then
  python3 - <<PY
import numpy as np, pathlib
def load(path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
ref = "${REF_EMB}"
pet = "${PETASE_EMB}"
out = pathlib.Path("${MERGED_EMB}")
emb = load(ref)
emb.update(load(pet))
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, **emb)
print("Wrote", out, "n", len(emb))
PY
fi

if [ ! -f "$MERGED_SEQ_DB" ]; then
  python3 - <<PY
import csv, pathlib
out = pathlib.Path("${MERGED_SEQ_DB}")
rows = {}
for path in ["${REF_SEQ_DB}", "${PETASE_SEQ_CSV}"]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = row.get("id","").strip()
            seq = row.get("sequence","").strip()
            if rid and seq:
                rows[rid] = seq
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","sequence"])
    w.writeheader()
    for rid, seq in rows.items():
        w.writerow({"id": rid, "sequence": seq})
print("Wrote", out, "rows", len(rows))
PY
fi

EXTRA_PROTECTED_FLAG=()
if [ -n "$EXTRA_PROTECTED" ]; then
  EXTRA_PROTECTED_FLAG=(--extra-protected "$EXTRA_PROTECTED")
fi

PYTHONPATH=. python3 scripts/generate_candidates_fasta.py \
  --parent-fasta "$WT_FASTA" \
  --mode retrieval \
  --embeddings "$MERGED_EMB" \
  --sequence-db "$MERGED_SEQ_DB" \
  --proposal-mode "$PROPOSAL_MODE" \
  --proposals "$PROPOSALS" \
  --proposals-per-neighbor "$PROPOSALS_PER_NEIGHBOR" \
  --trust-min "$TRUST_MIN" \
  --trust-max "$TRUST_MAX" \
  --seed "$SEED" \
  "${EXTRA_PROTECTED_FLAG[@]}" \
  --output "$CAND_FASTA"

ESM_FLAGS=()
if [ -n "$ESM_MODEL" ]; then
  ESM_FLAGS=(--model "$ESM_MODEL")
fi

PYTHONPATH=. python3 -m src.utils.esm_embed \
  "$CAND_FASTA" \
  -o "$CAND_EMB" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  "${ESM_FLAGS[@]}"

AF2_BIN_FLAG=()
if [ -n "$AF2_BIN" ]; then
  AF2_BIN_FLAG=(--af2-bin "$AF2_BIN")
fi

PYTHONPATH=. python3 scripts/surrogate_af2_workflow.py \
  --candidates "$CAND_FASTA" \
  --embeddings "$CAND_EMB" \
  --model "$MODEL_XGB" \
  --top-k "$TOP_K" \
  --direction max \
  --out-dir "$OUT_DIR" \
  "${AF2_BIN_FLAG[@]}" \
  $( [ "$RUN_AF2" = "1" ] && echo "--run-af2" )

echo "Overnight pipeline finished. AF2 command: ${OUT_DIR}/af2/colabfold_command.txt"
