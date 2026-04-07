#!/bin/bash
# Run from the scripts directory:
#   cd /path/to/scripts
#   sbatch submit_pipeline.sh

#SBATCH --job-name=jacobi_pipeline
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM_SUBMIT_DIR is the directory from which sbatch was called.
# cd there so all relative paths in main.py resolve correctly.
# ---- adapt module loads to your cluster ----
module purge
module load gcc openmpi cmake python nvhpc
source my_science_env/bin/activate
python3 main.py

