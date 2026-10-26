#!/bin/bash
#SBATCH --job-name=skgame-classical
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Account and partition are given on the command line (see submit_all.sh):
#   sbatch -A $SKGAME_ACCOUNT -p $SKGAME_PARTITION cineca/job_classical.sbatch
# One array task per problem class (`python main_simulation.py plan` lists them).
# Resumable: resubmitting skips every (class, N) already written.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cineca/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python main_simulation.py classical --out "$SKGAME_OUT" --task "$SLURM_ARRAY_TASK_ID"
