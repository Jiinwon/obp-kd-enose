#!/usr/bin/env bash
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -J gnina_array
#SBATCH -o gnina_array_%A_%a.out
#SBATCH -e gnina_array_%A_%a.err

# Example GNINA array job. It loads the CUDA module, activates the gnina
# environment, and runs docking for a grid of OBP/VOC combinations.
set -eE -o pipefail
source /etc/profile || true
module purge || true
module load cuda/12.0.0
set +u
source ~/anaconda3/bin/activate gnina_env
set -u
GNINA="$HOME/src/gnina/build/bin/gnina"

# Lists of proteins and VOCs. Extend as needed.
OBP_LIST=(obp1 obp2)
VOC_LIST=(vocA vocB)

# Map the SLURM array index to an OBP/VOC pair
TOTAL_OBP=${#OBP_LIST[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
obp_idx=$((TASK_ID % TOTAL_OBP))
voc_idx=$((TASK_ID / TOTAL_OBP))
OBP=${OBP_LIST[$obp_idx]}
VOC=${VOC_LIST[$voc_idx]}

# Prepare output directory
OUT_DIR="docking/outputs/${VOC}/${OBP}"
mkdir -p "$OUT_DIR"

# Run GNINA
"$GNINA" -r "docking/inputs/${OBP}.pdbqt" \
         -l "docking/inputs/${VOC}.pdbqt" \
         --center_x 0 --center_y 0 --center_z 0 \
         --size_x 10 --size_y 10 --size_z 10 \
         --out "$OUT_DIR/result.sdf" \
         --log "$OUT_DIR/result.json"

set +u
conda deactivate >/dev/null 2>&1 || true
