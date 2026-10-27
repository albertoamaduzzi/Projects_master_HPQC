# Running everything on your own computer

The whole campaign also runs on a laptop or workstation (Linux, macOS, or Windows through WSL): the classical part uses your CPU, and the QPU part only needs the Leap token and internet. For Leonardo see [README_CINECA.md](README_CINECA.md); the two can be mixed (`cineca/sync.sh`).

Measured times on 2–4 cores (this repository's test machine):

| preset | N grid | classical + game | simulated annealers | total |
|---|---|---|---|---|
| `smoke` | 8, 12, 16, 20 | ~10 s | ~5 s | **~20 s** |
| `small` | 8, 12, …, 64 | < 1 min (3 s per class at N = 64) | ~2 min (SA on the embedded problems dominates) | **~3 min** |
| `full` | 8, 16, …, 160 | ~2–2.5 h per problem class (4 classes, in parallel up to `JOBS`) | minutes | **~3 h with 4 cores** |

## 1. Get the code

```bash
git clone https://github.com/albertoamaduzzi/Projects_master_HPQC.git
cd Projects_master_HPQC/scripts/QuantumComputing/Dimod/src/unlearnable_games_explanation
```
(later: `git pull`)

## 2. Environment (once)

```bash
bash local/setup_env.sh               # creates .venv, installs cineca/requirements.txt, runs the 107 tests
WITH_NB=1 bash local/setup_env.sh     # alternatively: + Jupyter, for the notebook
source .venv/bin/activate             # in every new terminal
```
It needs Python ≥ 3.10 (`python3 --version`); choose another interpreter with `PYTHON=python3.12 bash local/setup_env.sh`. I tested the script on a clean copy: it installed everything and the tests passed.

## 3. D-Wave token (only for the real QPU)

How to obtain it (Leap account, Cineca ISCRA-C request, copying the token): [`../../LEAP_TOKEN.md`](../../LEAP_TOKEN.md).

```bash
dwave config create          # paste the token of the Cineca Leap project
dwave solvers -la            # an Advantage2_* (Zephyr) and an Advantage_* (Pegasus) should be listed
```

## 4. Run the pipeline

```bash
bash local/run_pipeline.sh                                    # smoke, simulated annealers, ~20 s
PRESET=small bash local/run_pipeline.sh                       # ~3 min
PRESET=small ANNEALER=qpu QPU_CALLS=2 bash local/run_pipeline.sh   # first real QPU test: 2 calls per topology
PRESET=full  ANNEALER=both bash local/run_pipeline.sh         # production: simulated + QPU, Zephyr and Pegasus
```

`run_pipeline.sh` runs the four stages in order. Each one is resumable, so after an interruption the same command continues where it stopped:

1. `init`: writes `results/<preset>/config.json` (back-ends `qpu:zephyr:clique qpu:pegasus:clique sim:zephyr:clique sim:pegasus:clique`) and all instances;
2. `classical`: one process per problem class, `JOBS` at a time; logs go to `results/<preset>/logs/`;
3. annealer:
   * `ANNEALER=sim`: `sim:logical` plus the embedded Zephyr/Pegasus problems solved by classical SA;
   * `ANNEALER=qpu`: `qpu --check` (no QPU time spent), then both QPU back-ends;
   * `ANNEALER=both`: both of the above;
4. `analyze`: `summary/summary.csv`, `backend_comparison.csv` and the figures.

| variable | default | meaning |
|---|---|---|
| `PRESET` | `smoke` | `smoke`, `small`, `full` |
| `OUT` | `results/$PRESET` | campaign directory |
| `ANNEALER` | `sim` | `sim`, `qpu`, `both`, `none` |
| `JOBS` | min(#CPU, 4) | problem classes computed in parallel |
| `QPU_CALLS` | unlimited | stop after this many QPU calls per back-end |
| `INIT_ARGS` | | extra `init` options, e.g. `"--Ns 8:96:8 --num-reads 500 --chain-strength-prefactor 1.0"` |

The same stages by hand, if you want to drive them yourself:
```bash
python main_simulation.py init --preset small --out results/small
python main_simulation.py classical --out results/small --task 0      # ... --task 3
python main_simulation.py sim --out results/small --backend sim:logical sim:zephyr:clique sim:pegasus:clique
python main_simulation.py qpu --out results/small --check
python main_simulation.py qpu --out results/small
python main_simulation.py analyze --out results/small
```

## 4b. Spending QPU time safely (budget guard)

Every QPU command accepts `--budget-seconds S`. Before each call it predicts the cost with the solver's timing model, times a safety factor ≥ 1.5. It makes the call only if the total spent stays ≤ S, and it counts the **real** charge after each call. `--spent` prints what has been charged so far. `--check` prints the prediction and spends nothing. Details: [`../../dwave_in_the_code.pdf`](../../dwave_in_the_code.pdf).

**Test with at most 1 second of QPU time** (both topologies, N = 8 and 16, one instance, 100 reads: 4 calls, normally ~0.1–0.2 s):
```powershell
# PowerShell (Windows). In bash write T=results/qpu_test instead of the first line.
$T = "results\qpu_test"
python main_simulation.py init --preset smoke --out $T --Ns 8,16 --n-instances 1 --regimes sg_satisficing --num-reads 100
python main_simulation.py qpu --out $T --check
python main_simulation.py qpu --out $T --budget-seconds 1
python main_simulation.py qpu --out $T --spent
```

**Full campaign within a monthly budget** (check on 27/10: Zephyr 52.3 s + Pegasus 105.0 s ≈ 157 s at 1000 reads):
```powershell
$F = "results\full"
python main_simulation.py init --preset full --out $F --force      # add --num-reads 500 to halve the cost (bash: F=results/full)
python main_simulation.py qpu --out $F --check
python main_simulation.py qpu --out $F --budget-seconds 170        # never passes 170 s; rerun continues
python main_simulation.py qpu --out $F --spent
```

## 5. Look at the results

```bash
ls results/small/summary/          # summary.csv, backend_comparison.csv, *.png
jupyter lab main_simulation.ipynb  # set SKGAME_OUT=results/small SKGAME_SAMPLER=none SKGAME_RUN_CLASSICAL=0 to only analyse
```

## 6. The explanatory figures

```bash
cd ../visualization
python SK_game/sk_game_explanation.py
python optimal_algorithms/algorithms_explanation.py
python embedding_explanation/embedding_explanation.py
python noise_and_chains/noise_and_chains.py
bash explanation_project/build.sh      # -> Dimod/explanation_project.pdf
```
