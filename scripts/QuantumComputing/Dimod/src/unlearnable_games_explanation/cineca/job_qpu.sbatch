#!/bin/bash
#SBATCH --job-name=skgame-qpu
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# lrd_all_serial runs on the login nodes (outbound HTTPS) and is not charged to the CPU budget.
# Only QPU time is spent. Resumable: resubmit after a timeout and it continues.
# Run cineca/check_internet.sh first; if lrd_all_serial has no internet use run_qpu_login.sh.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cineca/env.sh
python main_simulation.py qpu --out "$SKGAME_OUT" ${SKGAME_QPU_ARGS:-}
