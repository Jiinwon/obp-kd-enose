#!/bin/bash
#SBATCH --job-name=train_teacher

source env_activate.sh
python src/train/train_teacher.py --config configs/train_teacher.yaml
