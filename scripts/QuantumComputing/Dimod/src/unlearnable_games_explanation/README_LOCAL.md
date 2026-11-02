# Running everything on your own computer

> **For the QPU runs, follow [PIPELINE.md](PIPELINE.md).** It is one linear path: list of N → calibration of p_opt(N, machine) → SK runs → ground states and energies. This page covers installation and every other option.

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
bash local/setup_env.sh               # creates .venv, installs cineca/requirements.txt, runs the 128 tests
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

## 4. Which commands, for which setting

All commands run from this folder (`unlearnable_games_explanation`). The examples are PowerShell; in bash write `F=results/sk` instead of `$F = "results\sk"`, and `/` in paths.

The D-Wave tools live in the package **`dwave_utils/`**:

| module | what it does | command |
|---|---|---|
| `dwave_utils/feasibility.py` | yes / marginal / no: can a QPU represent the problem faithfully? | `feasibility` |
| `dwave_utils/pipeline.py` | the QPU pipeline: budget → calibrate → run → report → compare | `budget`, `calibrate`, `run`, `report`, `compare` |
| `dwave_utils/coupling_scan.py` | the chain-strength scan behind `calibrate` | `coupling-scan` |
| `dwave_utils/shimming.py` | flux-bias shimming of a QPU | `shim` |
| `dwave_utils/noise_fit.py` | σ and β_eff measured from the QPU's own small-N reads | `noise-fit` |

Every command can be launched in two equivalent ways:
```powershell
python main_simulation.py feasibility --topology zephyr --N 16 64 104
python -m dwave_utils     feasibility --topology zephyr --N 16 64 104
```
Each command takes `--help`. Each step is resumable: rerunning a command skips what is already saved.

### 4.1 Feasibility check: no QPU, no token, no campaign
Before spending anything, check whether the QPU can hold the problem faithfully:
```powershell
python -m dwave_utils feasibility --topology zephyr  --N 16 32 64 104                 # pure SK
python -m dwave_utils feasibility --topology pegasus --N 16 64 104 160 --J0 3         # ferromagnetic class
python -m dwave_utils feasibility --topology zephyr  --N 64 104 --extended-j-range --sigma 0.01 --plot feas.png
python -m dwave_utils feasibility --topology zephyr  --N 64 --J-file results\sk\instances\eps0_J00\N0064.npz
```
It prints, per N: chain length, the chain coupling the fields need (p_low), the relative control error, the temperature relative to T_c, the σ and β_eff a YES would need, and the verdict. σ (`--sigma`, default 0.02) and β_eff (`--beta`, default 5) are assumed values, not measurements. YES is necessary, not sufficient (see `dwave_utils/feasibility.py`).

### 4.2 Everything simulated: no token
The whole pipeline, with classical simulated annealing standing in for the QPU (bash or WSL):
```bash
bash local/run_pipeline.sh                    # smoke preset, ~20 s
PRESET=small bash local/run_pipeline.sh       # N = 8..64, ~3 min
```
| variable | default | meaning |
|---|---|---|
| `PRESET` | `smoke` | `smoke`, `small`, `full` |
| `OUT` | `results/$PRESET` | campaign directory |
| `ANNEALER` | `sim` | `sim`, `qpu`, `both`, `none` |
| `JOBS` | min(#CPU, 4) | problem classes computed in parallel |
| `QPU_CALLS` | unlimited | stop after this many QPU calls per back-end |
| `INIT_ARGS` | | extra `init` options, e.g. `"--Ns 8:96:8 --num-reads 500"` |

The same stages by hand, which also work in PowerShell:
```powershell
python main_simulation.py init --preset small --out results\small
python main_simulation.py classical --out results\small
python main_simulation.py sim --out results\small --backend sim:logical sim:zephyr:clique sim:pegasus:clique
python main_simulation.py analyze --out results\small
```

### 4.3 First contact with the QPU: at most 1 second
Both topologies, N = 8 and 16, one instance, 100 reads. That is 4 calls, normally about 0.1–0.2 s:
```powershell
$T = "results\qpu_test"
python main_simulation.py init --preset smoke --out $T --Ns 8,16 --n-instances 1 --regimes sg_satisficing --num-reads 100
python main_simulation.py qpu --out $T --check                 # estimate, spends nothing
python main_simulation.py qpu --out $T --budget-seconds 1
python main_simulation.py qpu --out $T --spent
```

### 4.4 The QPU pipeline: calibrate, then run (the main setting)
`--total` is the QPU time left in your Leap account **before** you start (Leap dashboard; ≈ 178 s). Use the same value at every step: each command subtracts what the campaign has already spent.

**Setting A: all 4 problem classes, the whole budget.**
```powershell
$F = "results\sk"
python -m dwave_utils feasibility --topology zephyr --N 16 64 104          # optional sanity check, no QPU
python main_simulation.py init --preset full --out $F --Ns 8:160:8 --n-instances 5
python -m dwave_utils budget    --out $F --total 178                       # 0. predicted cost, spends nothing
python -m dwave_utils calibrate --out $F --budget-seconds 15               # 1. p_opt per (machine, N), ~9 s
python -m dwave_utils budget    --out $F --total 178 --fill                # 1b. sets the reads so the run spends the rest
python -m dwave_utils run       --out $F --check                           #    estimate only
python -m dwave_utils run       --out $F --total 178                       # 2. every (class, N, machine) with its p_opt
python -m dwave_utils report    --out $F                                   # 3. results\sk\results\sk_qpu_results.csv
```

**Setting B: pure SK only.** Same commands, with `init` restricted to one class. It costs about ¼ of A at the same reads; `--fill` then gives it about 4× more reads per call:
```powershell
python main_simulation.py init --preset full --out results\sk_only --Ns 8:160:8 --n-instances 5 --regimes sg_satisficing
```

**Setting C: fewer sizes.** Any list of N works, e.g. `--Ns 16,32,64,104` or `--Ns 8:104:8`. Zephyr embeds up to N = 104; above that only Pegasus runs (up to about 177).

**Calibration options.**
- There is one p_opt per (machine, N), shared by every class. It is measured on the pure SK class; choose another with `calibrate --problem eps0_J03`.
- `--threshold 0.01` accepts up to 1 % broken chains: this gives a smaller p and less compression.
- Output: `results\sk\calibration\p_opt.csv` and `p_opt.png`. The table is stored in `config.json`, and `run` refuses to start if an (N, machine) is missing.

### 4.5 Compare: QPU planner vs classical planners vs the game (no QPU)
```powershell
python main_simulation.py classical --out $F        # CPU only: classical solvers, the game, references (hours for the full preset)
python -m dwave_utils compare --out $F              # -> results\sk\compare\compare.csv, compare.png, reference_quality.png
```
Problem classes go from easy to hard. The metrics:
- residual of the best and of the typical sample;
- P(exact) per sample, meaningful for N ≤ 24;
- P(residual ≤ 1 %) per sample: **satisficing success**, the main metric at large N;
- time to target, TTT99.

Above N = 24 the reference is the best known one (grey band in the figure); `reference_quality.png` checks it against the Parisi curve. Details: [PIPELINE.md](PIPELINE.md).

### 4.6 Optional QPU settings
**Extended coupler range.** Chains can use −2, which halves the compression. Make a separate campaign so the two can be compared:
```powershell
python main_simulation.py init --preset full --out results\sk_ext --Ns 64,104 --n-instances 5 --extended-j-range
```
**Shimming.** Measures and corrects the qubits' flux biases with D-Wave's tutorial procedure. Run it after `calibrate` (it uses p_opt) and before `run`. It costs about 1.5–2 s per machine:
```powershell
python -m dwave_utils shim --out $F --backend qpu:zephyr:clique  --N 104 --check
python -m dwave_utils shim --out $F --backend qpu:zephyr:clique  --N 104 --iterations 30 --budget-seconds 2
python -m dwave_utils shim --out $F --backend qpu:pegasus:clique --N 160 --iterations 30 --budget-seconds 2
#   -> results\sk\shims\<backend>\fbos.json, history.json, shim_history.png; later `run` calls use the offsets
```
Options:
- `--alpha-phi 0`: only measure the biases, change nothing;
- `--level chain`: Teza et al.'s per-chain variant;
- `--no-gauge`: no random spin-reversal gauge per iteration;
- `--no-apply`: save the offsets without using them.

In `shim_history.png` the magnetisation should fall to the dashed line 1/√reads; anything above that line is bias.

**Manual chain-strength scan** (what `calibrate` does, with every knob exposed):
```powershell
python -m dwave_utils coupling-scan --out $F --Ns 8:48:8 104 --prefactors 0.75 1.0 1.414 2.0 3.0 4.0 6.0 --check
python -m dwave_utils coupling-scan --out $F --Ns 8:48:8 104 --budget-seconds 7
python -m dwave_utils coupling-scan --out $F --summary-only        # recompute the choice from saved results
```

### 4.6b Measure the machine: σ and β_eff (no QPU)
After a run that includes small N (N = 8 is enough; N ≤ 12 is used by default), fit the control error σ and the effective inverse temperature β_eff of each machine from its own reads, then use them in the feasibility check:
```powershell
python -m dwave_utils noise-fit   --out $F                       # -> results\sk\calibration\noise_fit.json, noise_fit.png
python -m dwave_utils noise-fit   --out $F --max-N 16            # also N = 16 (slower: minutes per call)
python -m dwave_utils feasibility --topology zephyr  --N 16 64 104 --from-fit $F
python -m dwave_utils feasibility --topology pegasus --N 16 64 104 160 --from-fit $F
```
It uses only reads with no broken chain (`--include-broken` to use all of them). The model and its limits are in the docstring of `dwave_utils/noise_fit.py`.

### 4.7 From Python (notebook)
```python
import dwave_utils as du
du.check_feasibility("zephyr", [16, 64, 104])
out, cfg = du.open_campaign("results/sk")
du.budget(out, cfg, total=178)
du.calibrate(out, cfg, budget_s=15)
du.budget(out, cfg, total=178, fill=True)
du.run(out, cfg, budget_s=du.left(out, total=178))
rows = du.report(out, cfg)
rows = du.compare(out, cfg)
du.shim(out, cfg, "qpu:zephyr:clique", N=104, iterations=30)
du.fit_noise(out, cfg)                                   # sigma, beta_eff per machine
m = du.load_fit(out, "zephyr")
du.check_feasibility("zephyr", [16, 64, 104], sigma=m["sigma"], beta=m["beta_eff"])
```
The submodules stay importable as well: `from dwave_utils import pipeline, coupling_scan, feasibility, shimming`.

### 4.8 Checks and bookkeeping (no QPU)
```powershell
python main_simulation.py qpu --out $F --spent      # QPU time charged so far, per machine (main runs)
python main_simulation.py validate --out $F         # recompute every stored number from the stored inputs
python main_simulation.py ground --out $F           # ground state of every instance, its source, certified or not
python main_simulation.py chains --out $F           # solution quality vs number of broken chains
python main_simulation.py analyze --out $F          # summary.csv and figures
python main_simulation.py plan --out $F             # sizes, classes, p used for each N
```
Every field these files contain is described in [RESULTS_FORMAT.md](RESULTS_FORMAT.md). If a QPU's working graph has changed since the embeddings were cached, the code re-checks each cached embedding and recomputes it before using any QPU time.

### 4.9 If a run stops

Every step is resumable. Rerun the **same command** and it continues: finished calls are saved and never repeated, so no QPU time is spent twice. `python -m dwave_utils run --out $F --check` shows what is left and its cost.

**`PermissionError: [WinError 5] ... busclique ... .lru~ -> .lru` (Windows).** minorminer's clique cache in `.venv\data` was locked by OneDrive or an antivirus. The current code retries and then falls back to computing without the cache, so this no longer stops a run; see §12 of [`../../LESSONS_all_to_all_annealing.md`](../../LESSONS_all_to_all_annealing.md). To avoid it altogether, pause OneDrive during runs, or keep the repository and `.venv` outside OneDrive. If a stale `.lru~` file is left in `.venv\data\busclique_7\clique\`, delete it.

**`Unlink of file ... failed. Should I try again?` (git on Windows).** The file is open, for example a PDF in a viewer, or being synced. Close it and answer `y`.

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
