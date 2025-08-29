# obp-kd-enose

Scaffold for an e-nose ML/DL pipeline combining docking priors with low-cost
sensor signals. This repository provides placeholders for data processing,
model training, knowledge distillation, and user-side inference.

## Config-first workflow
All paths and hyperparameters are stored in YAML files under `configs/`.
Developers modify these YAMLs and launch training via SLURM scripts in `hpc/`.
End users run a single script (`scripts/user_run_local.sh`) which sets up a
lightweight environment and calls the demo using `configs/user_infer.yaml`.
