# Sourced by every job and by you on the login node:  source cineca/env.sh
# Edit the four variables below once; everything else derives from them.
export SKGAME_ACCOUNT="${SKGAME_ACCOUNT:-CHANGE_ME}"               # your Slurm account (saldo -b)
export SKGAME_PARTITION="${SKGAME_PARTITION:-dcgp_usr_prod}"       # CPU partition for the classical stage
export SKGAME_VENV="${SKGAME_VENV:-$WORK/venvs/skgame}"            # virtual environment
export SKGAME_OUT="${SKGAME_OUT:-$WORK/skgame_results/full}"       # campaign directory (one per campaign)

module load python 2>/dev/null || module load python/3.11
source "$SKGAME_VENV/bin/activate"
# directory holding main_simulation.py (this file lives in its cineca/ subfolder)
export SKGAME_PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NUMBA_CACHE_DIR="$SKGAME_PKG/.numba_cache"
export PYTHONUNBUFFERED=1
