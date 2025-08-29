# obp-kd-enose

Scaffold for an e-nose ML/DL pipeline combining docking priors with low-cost
sensor signals. This repository provides placeholders for data processing,
model training, knowledge distillation, and user-side inference.

## Quickstart
### Developers (HPC)
1. Edit YAML configs under `configs/`.
2. Submit SLURM jobs:
   ```bash
   sbatch hpc/train_teacher.sbatch
   sbatch hpc/train_student.sbatch
   ```

### Users (local CPU)
Run the single helper script which creates a venv and launches the demo:
```bash
bash scripts/user_run_local.sh --input sample.csv
```

## Config-first workflow
All paths and hyperparameters are stored in YAML files under `configs/`.
Developers modify these YAMLs and launch training via SLURM scripts in `hpc/`.
End users run a single script (`scripts/user_run_local.sh`) which sets up a
lightweight environment and calls the demo using `configs/user_infer.yaml`.
