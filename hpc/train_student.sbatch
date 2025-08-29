#!/bin/bash
#SBATCH --job-name=train_student

source env_activate.sh
python src/train/train_student.py --config configs/train_student.yaml
