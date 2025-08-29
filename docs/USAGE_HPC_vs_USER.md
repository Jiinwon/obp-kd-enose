# Usage

## Developers (HPC)
- Edit YAML configs under `configs/` to set paths and hyperparameters.
- Submit SLURM jobs with the helper scripts in `hpc/` (e.g. `sbatch hpc/train_teacher.sbatch`).
- Training code reads all settings from the provided YAML files.

## Users (no HPC)
- Run `scripts/user_run_local.sh`.
- The script creates a virtual environment, installs `requirements_user.txt`,
  downloads a release bundle containing a student ONNX model and
  `configs/user_infer.yaml`, and executes the demo:
  `python -m src.infer.user_demo --config configs/user_infer.yaml`.
