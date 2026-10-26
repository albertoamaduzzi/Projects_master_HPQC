#!/bin/bash
# Local environment (Linux, macOS, WSL). Run from the unlearnable_games_explanation folder:
#   bash local/setup_env.sh            # core: numpy scipy numba matplotlib dimod dwave-* pytest
#   WITH_NB=1 bash local/setup_env.sh  # + jupyter, to run main_simulation.ipynb
# Creates .venv here; activate it afterwards with:  source .venv/bin/activate
set -eo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 needed"'
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r cineca/requirements.txt
if [ -n "${WITH_NB:-}" ]; then
    python -m pip install -r cineca/requirements-nb.txt jupyterlab
fi
python -m pytest -q
echo
echo "environment ready: $(pwd)/.venv  ($(python --version))"
echo "next: source .venv/bin/activate && bash local/run_pipeline.sh"
