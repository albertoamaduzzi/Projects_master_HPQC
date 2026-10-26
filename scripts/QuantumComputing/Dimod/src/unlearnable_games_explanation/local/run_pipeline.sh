#!/bin/bash
# Whole campaign on this computer: init -> classical (+game) -> annealer -> analysis.
# Every stage is resumable: rerun the same command after an interruption.
#
#   bash local/run_pipeline.sh                               # smoke preset, simulated annealer, ~1 min
#   PRESET=small ANNEALER=sim bash local/run_pipeline.sh     # N = 8..64, tens of minutes
#   PRESET=full  ANNEALER=qpu bash local/run_pipeline.sh     # needs the Leap token (dwave config create)
#
# Variables:
#   PRESET    smoke | small | full                 (default smoke)
#   OUT       campaign directory                   (default results/<PRESET>)
#   ANNEALER  sim | qpu | both | none              (default sim; sim runs logical + embedded Zephyr/Pegasus SA)
#   JOBS      problem classes computed in parallel (default: number of CPUs, at most 4)
#   QPU_CALLS stop the QPU stage after this many calls per back-end (default: no limit)
#   INIT_ARGS extra arguments for `init`, e.g. "--Ns 8:96:8 --num-reads 500"
set -eo pipefail
cd "$(dirname "$0")/.."
[ -n "${VIRTUAL_ENV:-}" ] || { [ -f .venv/bin/activate ] && source .venv/bin/activate; }
PRESET="${PRESET:-smoke}"
OUT="${OUT:-results/$PRESET}"
ANNEALER="${ANNEALER:-sim}"
CPUS=$(python -c 'import os; print(os.cpu_count() or 1)')
JOBS="${JOBS:-$(( CPUS < 4 ? CPUS : 4 ))}"
mkdir -p "$OUT/logs"
t0=$(date +%s)

echo "== 1/4 init ($PRESET -> $OUT)"
BACKENDS="qpu:zephyr:clique qpu:pegasus:clique sim:zephyr:clique sim:pegasus:clique"
python main_simulation.py init --preset "$PRESET" --out "$OUT" --backends $BACKENDS ${INIT_ARGS:-} | tail -4
python main_simulation.py instances --out "$OUT"

echo "== 2/4 classical solvers + game ($JOBS in parallel; logs in $OUT/logs)"
NCLS=$(python -c "
import sys; sys.argv=['x']; import main_simulation as ms
print(len(ms._classes(ms.load_config(__import__('pathlib').Path('$OUT')))))")
for i in $(seq 0 $((NCLS - 1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
    python main_simulation.py classical --out "$OUT" --task "$i" > "$OUT/logs/classical_$i.log" 2>&1 &
done
wait
tail -n 1 "$OUT"/logs/classical_*.log

echo "== 3/4 annealer ($ANNEALER)"
MAXC=${QPU_CALLS:+--max-calls $QPU_CALLS}
if [ "$ANNEALER" = sim ] || [ "$ANNEALER" = both ]; then
    python main_simulation.py sim --out "$OUT" --backend sim:logical sim:zephyr:clique sim:pegasus:clique \
        | grep -v '^\[anneal\] *[0-9]' || true
fi
if [ "$ANNEALER" = qpu ] || [ "$ANNEALER" = both ]; then
    python main_simulation.py qpu --out "$OUT" --check
    python main_simulation.py qpu --out "$OUT" $MAXC | tee "$OUT/logs/qpu.log" | grep -v '^\[anneal\] *[0-9]' || true
fi

echo "== 4/4 analysis"
python main_simulation.py analyze --out "$OUT"
echo "done in $(( $(date +%s) - t0 )) s. Results: $OUT/summary/"
