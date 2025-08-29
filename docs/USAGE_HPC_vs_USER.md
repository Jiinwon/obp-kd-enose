# Usage

## Developers (HPC)
- Edit YAML configs under `configs/` to set paths and hyperparameters.
- Submit SLURM jobs:
  ```bash
  sbatch hpc/train_teacher.sbatch
  sbatch hpc/train_student.sbatch
  ```

## Users (no HPC)
Run the demo on a local CPU machine:
```bash
bash scripts/user_run_local.sh --input sample.csv
```
