#!/usr/bin/env bash
set -euo pipefail

# Multimodel filter pipeline:
# 1) Start from a large candidate set (FASTA + surrogate predictions).
# 2) Keep top-N by surrogate residual.
# 3) Run sequence QC filters.
# 4) Run Rosetta ddG on the filtered set.
# 5) (Optional) merge AF2 QC metrics and filter by pLDDT.
# 6) Export final shortlist.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"

PARENT_FASTA="${PARENT_FASTA:-${ROOT_DIR}/data/processed/petase_wt.fasta}"
CAND_FASTA="${CAND_FASTA:-${ROOT_DIR}/data/processed/candidates_100k.fasta}"
PRED_CSV="${PRED_CSV:-${ROOT_DIR}/results/surrogate_af2/context_candidates_100k_predictions_petaseft.csv}"

TOPN="${TOPN:-500}"
MAX_MUT="${MAX_MUT:-5}"

WORK_DIR="${WORK_DIR:-${ROOT_DIR}/results/surrogate_af2/multimodel_filter}"
TOPN_FASTA="${TOPN_FASTA:-${WORK_DIR}/topn.fasta}"
TOPN_CSV="${TOPN_CSV:-${WORK_DIR}/topn.csv}"
QC_OUT="${QC_OUT:-${WORK_DIR}/topn_qc.csv}"
FILTERED_CSV="${FILTERED_CSV:-${WORK_DIR}/filtered_seq.csv}"
FILTERED_FASTA="${FILTERED_FASTA:-${WORK_DIR}/filtered_seq.fasta}"

ROSETTA_BIN="${ROSETTA_BIN:-}"
WT_PDB="${WT_PDB:-}"
ROSETTA_DB="${ROSETTA_DB:-${HOME}/rosetta/rosetta.binary.ubuntu.release-408/main/database}"
ROSETTA_OUT_DIR="${ROSETTA_OUT_DIR:-${WORK_DIR}/rosetta}"
ROSETTA_DDG_CSV="${ROSETTA_DDG_CSV:-${ROSETTA_OUT_DIR}/rosetta_ddg.csv}"
ROSETTA_DDG_WITH_WT="${ROSETTA_DDG_WITH_WT:-${ROSETTA_OUT_DIR}/rosetta_ddg_with_wt.csv}"
ROSETTA_DDG_MAX="${ROSETTA_DDG_MAX:-0.0}"

AF2_QC="${AF2_QC:-}" # optional QC with AF2 metrics (pLDDT)
PLDDT_MIN="${PLDDT_MIN:-90}"

FINAL_CSV="${FINAL_CSV:-${WORK_DIR}/final_shortlist.csv}"
FINAL_FASTA="${FINAL_FASTA:-${WORK_DIR}/final_shortlist.fasta}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing venv at ${VENV_PATH}. Set VENV_PATH or create the venv." >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"
mkdir -p "${WORK_DIR}"

echo "Selecting top-${TOPN} by surrogate residual..."
python - <<PY
import pandas as pd
from pathlib import Path

pred = Path("${PRED_CSV}")
topn_csv = Path("${TOPN_CSV}")
topn_fasta = Path("${TOPN_FASTA}")
in_fasta = Path("${CAND_FASTA}")

df = pd.read_csv(pred)
if "pred_residual" not in df.columns:
    raise SystemExit("pred_residual not found in surrogate predictions.")

df = df.sort_values("pred_residual", ascending=False).head(int("${TOPN}"))
topn_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(topn_csv, index=False)

ids = set(df["id"].astype(str))
with in_fasta.open() as f, topn_fasta.open("w") as out:
    write = False
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split()[0]
            write = seq_id in ids
        if write:
            out.write(line)
print(f"Wrote {topn_csv} and {topn_fasta}")
PY

echo "Running sequence QC on top-${TOPN}..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/qc_candidates.py" \
  --candidates-fasta "${TOPN_FASTA}" \
  --wt-fasta "${PARENT_FASTA}" \
  --predictions "${TOPN_CSV}" \
  --out "${QC_OUT}"

echo "Filtering by sequence QC (no protected mutations, no new glyco motifs, mut_count<=${MAX_MUT})..."
python - <<PY
import pandas as pd
from pathlib import Path

qc = pd.read_csv(Path("${QC_OUT}"))
pred = pd.read_csv(Path("${TOPN_CSV}"))[["id","pred_residual"]]
df = qc.merge(pred, on="id", how="left")

def is_false(x):
    if isinstance(x, bool):
        return not x
    return str(x).lower() in ("false","0","no","none","nan","")

mask = (
    (df["protected_mutations"].fillna(0) == 0) &
    (df["disulfide_seq_ok"] == True) &
    (df["mut_count"].fillna(9999) <= int("${MAX_MUT}")) &
    (df["glyco_motif_new"].apply(is_false)) &
    (df["duplicate_sequence"].apply(is_false)) &
    (df["in_training_seq"].apply(is_false))
)
filtered = df[mask].copy()

out_csv = Path("${FILTERED_CSV}")
out_csv.parent.mkdir(parents=True, exist_ok=True)
filtered.to_csv(out_csv, index=False)

ids = set(filtered["id"].astype(str))
in_fasta = Path("${TOPN_FASTA}")
out_fasta = Path("${FILTERED_FASTA}")
with in_fasta.open() as f, out_fasta.open("w") as out:
    write = False
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split()[0]
            write = seq_id in ids
        if write:
            out.write(line)
print(f"Wrote {out_csv} and {out_fasta} with {len(filtered)} candidates")
PY

if [[ -z "${ROSETTA_BIN}" || -z "${WT_PDB}" ]]; then
  echo "Skipping Rosetta: set ROSETTA_BIN and WT_PDB to run ddG."
  exit 0
fi

echo "Preparing Rosetta mutfiles..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/prepare_rosetta_mutfiles.py" \
  --candidates-fasta "${FILTERED_FASTA}" \
  --wt-fasta "${PARENT_FASTA}" \
  --out-dir "${ROSETTA_OUT_DIR}/mutfiles"

echo "Running Rosetta cartesian_ddg..."
PYTHONPATH=. python "${ROOT_DIR}/scripts/run_rosetta_cartesian_ddg.py" \
  --mutfiles-dir "${ROSETTA_OUT_DIR}/mutfiles" \
  --wt-pdb "${WT_PDB}" \
  --rosetta-bin "${ROSETTA_BIN}" \
  --out-csv "${ROSETTA_DDG_CSV}" \
  --out-dir "${ROSETTA_OUT_DIR}/out" \
  --jobs 16 \
  --resume \
  --extra-flags "-database ${ROSETTA_DB}"

echo "Extracting WT baseline and ddG from Rosetta .ddg files..."
python - <<PY
from pathlib import Path
import pandas as pd

rows = []
for path in Path(".").glob("IsPETaseWT_pool_*.ddg"):
    wt_vals, mut_vals = [], []
    with path.open() as f:
        for line in f:
            if " WT:" in line:
                parts = line.strip().split()
                if "WT:" in parts:
                    idx = parts.index("WT:")
                    wt_vals.append(float(parts[idx+1]))
            elif " MUT_" in line:
                parts = line.strip().split()
                mut_tok = next((p for p in parts if p.startswith("MUT_")), None)
                if mut_tok:
                    idx = parts.index(mut_tok)
                    mut_vals.append(float(parts[idx+1]))
    if wt_vals and mut_vals:
        wt = sum(wt_vals)/len(wt_vals)
        mut = sum(mut_vals)/len(mut_vals)
        rows.append({
            "id": path.stem,
            "wt_energy_reu": wt,
            "mut_energy_reu": mut,
            "ddg_reu": mut - wt,
        })

df = pd.DataFrame(rows)
out = Path("${ROSETTA_DDG_WITH_WT}")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
PY

echo "Merging surrogate + Rosetta and applying final filters..."
python - <<PY
import pandas as pd
from pathlib import Path

filtered = pd.read_csv(Path("${FILTERED_CSV}"))
ros = pd.read_csv(Path("${ROSETTA_DDG_WITH_WT}"))
merged = filtered.merge(ros, on="id", how="left")

# Optional AF2 QC merge
af2_path = Path("${AF2_QC}") if "${AF2_QC}" else None
if af2_path and af2_path.exists():
    af2 = pd.read_csv(af2_path)
    merged = merged.merge(af2[["id","plddt_mean","plddt_key_mean"]], on="id", how="left")

# Apply Rosetta threshold
merged = merged[merged["ddg_reu"] <= float("${ROSETTA_DDG_MAX}")].copy()

# Optional AF2 filter
if "plddt_mean" in merged.columns:
    merged = merged[merged["plddt_mean"] >= float("${PLDDT_MIN}")].copy()

out_csv = Path("${FINAL_CSV}")
out_csv.parent.mkdir(parents=True, exist_ok=True)
merged.sort_values("pred_residual", ascending=False).to_csv(out_csv, index=False)

# Export FASTA
ids = set(merged["id"].astype(str))
in_fasta = Path("${FILTERED_FASTA}")
out_fasta = Path("${FINAL_FASTA}")
with in_fasta.open() as f, out_fasta.open("w") as out:
    write = False
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split()[0]
            write = seq_id in ids
        if write:
            out.write(line)

print(f"Wrote {out_csv} and {out_fasta} with {len(merged)} candidates")
PY

echo "Multimodel filter pipeline complete."
