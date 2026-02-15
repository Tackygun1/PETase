#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline: generate 100k candidates, embed, score with context surrogate,
# run QC, and export top-k.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"

PARENT_FASTA="${PARENT_FASTA:-${ROOT_DIR}/data/processed/petase_wt.fasta}"
EMBEDDINGS_REF="${EMBEDDINGS_REF:-${ROOT_DIR}/data/processed/petase_pretrain_embeddings.npz}"
SEQUENCE_DB="${SEQUENCE_DB:-${ROOT_DIR}/data/processed/sequence_db_petase_pretrain.csv}"

PROPOSALS="${PROPOSALS:-100000}"
PROPOSALS_PER_NEIGHBOR="${PROPOSALS_PER_NEIGHBOR:-4}"
TRUST_MIN="${TRUST_MIN:-1}"
TRUST_MAX="${TRUST_MAX:-5}"
SEED="${SEED:-42}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-1}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
TOPK="${TOPK:-20}"

OUT_FASTA="${OUT_FASTA:-${ROOT_DIR}/data/processed/candidates_100k.fasta}"
CHUNK_DIR="${CHUNK_DIR:-${ROOT_DIR}/data/processed/candidates_100k_chunks}"
OUT_EMB="${OUT_EMB:-${ROOT_DIR}/data/processed/candidates_100k_esm.npz}"
PRED_OUT="${PRED_OUT:-${ROOT_DIR}/results/surrogate_af2/context_candidates_100k_predictions.csv}"
TOPK_CSV="${TOPK_CSV:-${ROOT_DIR}/results/surrogate_af2/context_top${TOPK}_100k.csv}"
TOPK_FASTA="${TOPK_FASTA:-${ROOT_DIR}/results/surrogate_af2/context_top${TOPK}_100k.fasta}"
QC_PRED="${QC_PRED:-${ROOT_DIR}/results/surrogate_af2/context_top${TOPK}_100k_predictions.csv}"
QC_OUT="${QC_OUT:-${ROOT_DIR}/results/surrogate_af2/context_top${TOPK}_100k_qc.csv}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/fireprot_context_xgb_dtm_q20.json}"
BASELINES_PATH="${BASELINES_PATH:-${ROOT_DIR}/models/fireprot_context_baselines.json}"

ROSETTA_BIN="${ROSETTA_BIN:-}"
WT_PDB="${WT_PDB:-}"
ROSETTA_OUT_DIR="${ROSETTA_OUT_DIR:-${ROOT_DIR}/results/surrogate_af2/rosetta_top${TOPK}_100k}"
ROSETTA_SCORE_CSV="${ROSETTA_SCORE_CSV:-${ROSETTA_OUT_DIR}/rosetta_ddg.csv}"
COMBINED_CSV="${COMBINED_CSV:-${ROOT_DIR}/results/surrogate_af2/context_top${TOPK}_100k_combined.csv}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing venv at ${VENV_PATH}. Set VENV_PATH or create the venv." >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"

PROTECTED_POSITIONS="$(
python - <<'PY'
try:
    from src.scoring.constraints import PROTECTED_POSITIONS
    print(",".join(str(x) for x in sorted(PROTECTED_POSITIONS)))
except Exception:
    print("160,206,237,87,161,185,203,239,273,289")
PY
)"

EXTRA_PROTECTED="${EXTRA_PROTECTED:-${PROTECTED_POSITIONS}}"

echo "Generating ${PROPOSALS} candidates..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/generate_candidates_fasta.py" \
  --parent-fasta "${PARENT_FASTA}" \
  --mode retrieval \
  --embeddings "${EMBEDDINGS_REF}" \
  --sequence-db "${SEQUENCE_DB}" \
  --proposals "${PROPOSALS}" \
  --proposals-per-neighbor "${PROPOSALS_PER_NEIGHBOR}" \
  --trust-min "${TRUST_MIN}" \
  --trust-max "${TRUST_MAX}" \
  --seed "${SEED}" \
  --extra-protected "${EXTRA_PROTECTED}" \
  --output "${OUT_FASTA}"

echo "Splitting FASTA into chunks of ${CHUNK_SIZE}..."
python - <<PY
from pathlib import Path

in_fasta = Path("${OUT_FASTA}")
out_dir = Path("${CHUNK_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

chunk_size = int("${CHUNK_SIZE}")
idx = 0
out = None
count = 0

with in_fasta.open() as f:
    for line in f:
        if line.startswith(">"):
            if count % chunk_size == 0:
                if out:
                    out.close()
                out = (out_dir / f"chunk_{idx:04d}.fasta").open("w")
                idx += 1
            count += 1
        if out:
            out.write(line)

if out:
    out.close()

print(f"Wrote {idx} chunks to {out_dir}")
PY

echo "Embedding chunks on ${EMBED_DEVICE}..."
for f in "${CHUNK_DIR}"/chunk_*.fasta; do
  out="${CHUNK_DIR}/$(basename "$f" .fasta).npz"
  PYTHONPATH=. python -m src.utils.esm_embed "$f" \
    -o "$out" \
    --device "${EMBED_DEVICE}" \
    --batch-size "${EMBED_BATCH_SIZE}"
done

echo "Merging embeddings..."
python - <<PY
from pathlib import Path
import numpy as np

chunk_dir = Path("${CHUNK_DIR}")
out = Path("${OUT_EMB}")
merged = {}
for npz in sorted(chunk_dir.glob("chunk_*.npz")):
    data = np.load(npz)
    merged.update({k: data[k] for k in data.files})
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, **merged)
print(f"Wrote {out} with {len(merged)} embeddings")
PY

echo "Scoring candidates with context surrogate..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/predict_xgb_context.py" \
  --embeddings "${OUT_EMB}" \
  --model "${MODEL_PATH}" \
  --baselines "${BASELINES_PATH}" \
  --out "${PRED_OUT}"

echo "Selecting top-${TOPK}..."
python - <<PY
import pandas as pd
from pathlib import Path

pred_path = Path("${PRED_OUT}")
topk_csv = Path("${TOPK_CSV}")
topk_fasta = Path("${TOPK_FASTA}")
in_fasta = Path("${OUT_FASTA}")

df = pd.read_csv(pred_path).sort_values("pred_residual", ascending=False)
top = df.head(int("${TOPK}")).copy()
top.to_csv(topk_csv, index=False)

ids = set(top["id"])
topk_fasta.parent.mkdir(parents=True, exist_ok=True)
with in_fasta.open() as f, topk_fasta.open("w") as out:
    write = False
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split()[0]
            write = seq_id in ids
        if write:
            out.write(line)

print(f"Wrote {topk_csv} and {topk_fasta}")
PY

echo "Preparing predictions for QC..."
python - <<PY
import pandas as pd
from pathlib import Path

src = Path("${TOPK_CSV}")
dst = Path("${QC_PRED}")
df = pd.read_csv(src)
df = df[["id", "pred_residual"]].rename(columns={"pred_residual": "prediction"})
dst.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dst, index=False)
print(f"Wrote {dst}")
PY

echo "Running QC checks (sequence + protected geometry)..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/qc_candidates.py" \
  --candidates-fasta "${TOPK_FASTA}" \
  --wt-fasta "${PARENT_FASTA}" \
  --train-seq-db "${SEQUENCE_DB}" \
  --predictions "${QC_PRED}" \
  --out "${QC_OUT}"

if [[ -n "${ROSETTA_BIN}" && -n "${WT_PDB}" && -f "${ROSETTA_BIN}" && -f "${WT_PDB}" ]]; then
  echo "Preparing Rosetta mutfiles..."
  MUTFILE_DIR="${ROSETTA_OUT_DIR}/mutfiles"
  PYTHONPATH=. python "${ROOT_DIR}/scripts/prepare_rosetta_mutfiles.py" \
    --candidates-fasta "${TOPK_FASTA}" \
    --wt-fasta "${PARENT_FASTA}" \
    --out-dir "${MUTFILE_DIR}"

  echo "Running Rosetta cartesian_ddg..."
  PYTHONPATH=. python "${ROOT_DIR}/scripts/run_rosetta_cartesian_ddg.py" \
    --mutfile-dir "${MUTFILE_DIR}" \
    --wt-pdb "${WT_PDB}" \
    --rosetta-bin "${ROSETTA_BIN}" \
    --out "${ROSETTA_SCORE_CSV}"

  echo "Combining surrogate + Rosetta scores..."
  python - <<PY
import pandas as pd
import numpy as np
from pathlib import Path

topk_csv = Path("${TOPK_CSV}")
rosetta_csv = Path("${ROSETTA_SCORE_CSV}")
out_csv = Path("${COMBINED_CSV}")

top = pd.read_csv(topk_csv)
ros = pd.read_csv(rosetta_csv)
df = top.merge(ros, on="id", how="left")

if "pred_residual" not in df.columns:
    raise SystemExit("Missing pred_residual in top-k CSV.")
if "ddg" not in df.columns:
    raise SystemExit("Missing ddg in Rosetta CSV.")

def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x * 0.0

df["z_pred"] = zscore(df["pred_residual"].astype(float))
df["z_ddg"] = zscore((-df["ddg"]).astype(float))  # lower ddg = better
df["combined_score"] = df["z_pred"] + df["z_ddg"]

out_csv.parent.mkdir(parents=True, exist_ok=True)
df.sort_values("combined_score", ascending=False).to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")
PY
else
  echo "Skipping Rosetta: set ROSETTA_BIN and WT_PDB to valid paths."
fi

echo "Done. Top-k: ${TOPK_CSV}"
echo "QC report: ${QC_OUT}"
