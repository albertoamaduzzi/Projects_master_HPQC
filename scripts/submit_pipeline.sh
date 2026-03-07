#!/bin/bash
# Submit from the login node:
#   sbatch submit_pipeline.sh
#
# Override any setting before calling sbatch, e.g.:
#   SLURM_PARTITION=debug SLURM_ACCOUNT=myproj sbatch submit_pipeline.sh

#SBATCH --job-name=jacobi_pipeline
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

# ---- adapt these module loads to your cluster ----
# module purge
# module load gcc openmpi cmake python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/main.py"

