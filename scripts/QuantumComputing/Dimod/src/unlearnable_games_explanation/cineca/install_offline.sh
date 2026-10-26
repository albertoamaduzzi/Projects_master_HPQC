#!/bin/bash
# Run on Leonardo. Creates the virtual environment from the uploaded wheelhouse: no internet used.
#   bash cineca/install_offline.sh $WORK/skgame_wheelhouse_py311.tar.gz
set -eo pipefail
ARCHIVE="${1:?usage: install_offline.sh <path to skgame_wheelhouse_pyXY.tar.gz>}"
VENV="${SKGAME_VENV:-$WORK/venvs/skgame}"
module load python 2>/dev/null || true
DIR="$(dirname "$ARCHIVE")/$(basename "$ARCHIVE" .tar.gz)"
[ -d "$DIR" ] || tar xzf "$ARCHIVE" -C "$(dirname "$ARCHIVE")"
want="$(cat "$DIR/PYTHON_VERSION")"
have="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [ "$want" != "$have" ]; then
    echo "wheelhouse built for Python $want but 'python' here is $have:"
    echo "load the matching module (module avail python) or rebuild with: build_wheelhouse.sh $have"
    exit 1
fi
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --find-links "$DIR" --upgrade pip
pip install --no-index --find-links "$DIR" -r "$DIR/requirements.lock"
echo "environment ready: $VENV  (python $have)"
