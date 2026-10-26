#!/bin/bash
#SBATCH --job-name=skgame-analyze
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cineca/env.sh
python main_simulation.py analyze --out "$SKGAME_OUT"
