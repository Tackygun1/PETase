#!/usr/bin/env bash
set -euo pipefail

# Batch AlphaFold runner for many single-sequence FASTAs.
# AlphaFold run_alphafold.py expects one sequence per FASTA.

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <input_fasta> <output_dir> <data_dir>" >&2
  exit 1
fi

INPUT_FASTA="$1"
OUTPUT_DIR="$2"
DATA_DIR="$3"

ALPHAFOLD_PY="${ALPHAFOLD_PY:-/home/gofish/alphafold/run_alphafold.py}"
CONDA_RUN="${CONDA_RUN:-/home/gofish/miniconda3/bin/conda run -n alphafold python}"
DB_PRESET="${DB_PRESET:-reduced_dbs}"
MAX_TEMPLATE_DATE="${MAX_TEMPLATE_DATE:-2024-12-31}"

SMALL_BFD="${SMALL_BFD:-${DATA_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta}"
UNIREF90="${UNIREF90:-${DATA_DIR}/uniref90/uniref90.fasta}"
MGNIFY="${MGNIFY:-${DATA_DIR}/mgnify/mgy_clusters_2022_05.fa}"
PDB70="${PDB70:-${DATA_DIR}/pdb70/pdb70}"
MMCIF_DIR="${MMCIF_DIR:-${DATA_DIR}/pdb_mmcif/mmcif_files}"
OBSOLETE_PDBS="${OBSOLETE_PDBS:-${DATA_DIR}/pdb_mmcif/obsolete.dat}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Splitting FASTA into single-sequence files..."
python - <<PY
from pathlib import Path

in_fasta = Path("${INPUT_FASTA}")
out_dir = Path("${TMP_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

current = None
seq = []
with in_fasta.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                (out_dir / f"{current}.fasta").write_text(f">{current}\\n{''.join(seq)}\\n")
            current = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if current:
        (out_dir / f"{current}.fasta").write_text(f">{current}\\n{''.join(seq)}\\n")
print(f"Wrote {len(list(out_dir.glob('*.fasta')))} files to {out_dir}")
PY

mkdir -p "${OUTPUT_DIR}"

for f in "${TMP_DIR}"/*.fasta; do
  id="$(basename "$f" .fasta)"
  if [[ -d "${OUTPUT_DIR}/${id}" ]]; then
    echo "Skipping ${id} (already exists)"
    continue
  fi
  echo "Running AlphaFold for ${id}..."
  ${CONDA_RUN} "${ALPHAFOLD_PY}" \
    --fasta_paths="$f" \
    --output_dir="${OUTPUT_DIR}" \
    --data_dir="${DATA_DIR}" \
    --db_preset="${DB_PRESET}" \
    --small_bfd_database_path="${SMALL_BFD}" \
    --uniref90_database_path="${UNIREF90}" \
    --mgnify_database_path="${MGNIFY}" \
    --template_mmcif_dir="${MMCIF_DIR}" \
    --obsolete_pdbs_path="${OBSOLETE_PDBS}" \
    --pdb70_database_path="${PDB70}" \
    --max_template_date="${MAX_TEMPLATE_DATE}" \
    --models_to_relax=none \
    --use_gpu_relax=false
done

echo "Batch AlphaFold complete."
