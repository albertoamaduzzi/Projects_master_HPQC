#!/bin/bash
# Run on YOUR machine (with internet). Downloads every wheel the campaign needs, built for
# Leonardo (Linux x86_64, glibc 2.28) and for the Python version of Leonardo's module, and
# packs them in one archive to upload. Works from Linux, macOS or WSL.
#
#   bash cineca/build_wheelhouse.sh 3.11          # Python version printed on Leonardo by:
#                                                 #   module load python; python --version
#   WITH_NB=1 bash cineca/build_wheelhouse.sh 3.11  # + jupyter nbconvert for the notebook job
set -eo pipefail
cd "$(dirname "$0")/.."
PYVER="${1:?usage: build_wheelhouse.sh <python version on Leonardo, e.g. 3.11>}"
NAME="skgame_wheelhouse_py${PYVER/./}"
rm -rf "$NAME"
reqs=(-r cineca/requirements.txt)
[ -n "${WITH_NB:-}" ] && reqs+=(-r cineca/requirements-nb.txt)
python3 -m pip download --dest "$NAME" --only-binary=:all: \
    --python-version "$PYVER" --implementation cp \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    "${reqs[@]}" pip
cp cineca/requirements*.txt "$NAME/"
# exact versions that were downloaded, so Leonardo installs exactly these
ls "$NAME"/*.whl | sed -E 's#.*/##; s#^([^-]+)-([^-]+)-.*#\1==\2#' | sort > "$NAME/requirements.lock"
echo "$PYVER" > "$NAME/PYTHON_VERSION"
tar czf "$NAME.tar.gz" "$NAME"
echo "built $NAME.tar.gz ($(du -h "$NAME.tar.gz" | cut -f1)), $(ls "$NAME"/*.whl | wc -l) wheels"
echo "upload:  scp $NAME.tar.gz <username>@login.leonardo.cineca.it:\$WORK/"
