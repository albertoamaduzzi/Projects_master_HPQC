#!/bin/bash
# Submit the whole campaign: classical array + QPU job, then the analysis once both finished.
#   bash cineca/submit_all.sh            # classical + qpu + analyze
#   bash cineca/submit_all.sh --no-qpu   # classical + analyze only
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$SKGAME_PKG"
mkdir -p logs
[ -f "$SKGAME_OUT/config.json" ] || { echo "run first: python main_simulation.py init --preset full --out $SKGAME_OUT"; exit 1; }
c=$(sbatch --parsable -A "$SKGAME_ACCOUNT" -p "$SKGAME_PARTITION" cineca/job_classical.sbatch)
echo "classical array: $c"
dep="afterany:$c"
if [ "${1:-}" != "--no-qpu" ]; then
    q=$(sbatch --parsable -A "$SKGAME_ACCOUNT" cineca/job_qpu.sbatch)
    echo "qpu job:         $q"
    dep="$dep:$q"
fi
a=$(sbatch --parsable -A "$SKGAME_ACCOUNT" --dependency="$dep" cineca/job_analyze.sbatch)
echo "analysis:        $a (starts when the others end)"
echo "follow with: squeue --me ; tail -f logs/*.out"
