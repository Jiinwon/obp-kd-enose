#!/usr/bin/env bash
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH -J enose_teacher
#SBATCH -o logs/teacher_%j.out
#SBATCH -e logs/teacher_%j.err
set -e
mkdir -p logs
source hpc/env_activate.sh
python -m src.train.train_teacher --config configs/train_teacher.yaml
