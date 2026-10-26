#!/bin/bash
#SBATCH --job-name=skgame-notebook
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Executes main_simulation.ipynb headless and saves the executed copy (with figures) next to the results.
# Parameters through the environment, e.g.
#   SKGAME_SAMPLER=none SKGAME_RUN_CLASSICAL=0 sbatch cineca/job_notebook.sbatch   # only analysis
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cineca/env.sh
export SKGAME_SAMPLER="${SKGAME_SAMPLER:-none}" SKGAME_RUN_CLASSICAL="${SKGAME_RUN_CLASSICAL:-0}"
jupyter nbconvert --to notebook --execute main_simulation.ipynb \
    --ExecutePreprocessor.timeout=-1 --output "$SKGAME_OUT/main_simulation_executed.ipynb"
