#!/usr/bin/env bash
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -J gnina_smoke
#SBATCH -o gnina_smoke_%j.out
#SBATCH -e gnina_smoke_%j.err
set -eE -o pipefail
export LC_ALL="${LC_ALL:-C}"
source /etc/profile || true
module purge || true
module load cuda/12.0.0
set +u
source ~/anaconda3/bin/activate gnina_env
set -u
cleanup() { set +u; conda deactivate >/dev/null 2>&1 || true; }
trap cleanup EXIT
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
GNINA="$HOME/src/gnina/build/bin/gnina"
echo "===== GPU / 라이브러리 점검 ====="
nvidia-smi || true
ldd "$GNINA" | egrep 'not found|libcuda|cudnn|torch|openbabel|molgrid' || true
"$GNINA" --version
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
WORKDIR="${SLURM_TMPDIR:-$PWD/gnina_smoke_${SLURM_JOB_ID:-$$}}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
cat > receptor.pdbqt <<'PQR'
REMARK  minimal receptor
ATOM      1  C1  UNL     1       0.000   0.000   0.000  0.00  0.00    -0.115 C
TER
PQR
cat > ligand.pdbqt <<'PQL'
REMARK  minimal ligand
ROOT
ATOM      1  C1  LIG     1       0.500   0.000   0.000  0.00  0.00    -0.100 C
ENDROOT
TORSDOF 0
PQL
echo "===== GNINA score_only ====="
set -x
"$GNINA" -r receptor.pdbqt -l ligand.pdbqt \
  --center_x 0 --center_y 0 --center_z 0 \
  --size_x 10 --size_y 10 --size_z 10 \
  --score_only
set +x
echo "===== 완료 ====="
pwd
ls -l
