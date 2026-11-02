# The QPU pipeline, step by step

**Input:** a list of N (number of agents) and the machines (Zephyr = Advantage2, Pegasus = Advantage).
**Output:** for every (machine, problem class, N, instance): the best configuration found (ground-state candidate), its energy, its reward R_N, and whether it equals the reference optimum.

```
list of N ──► 0. budget ──► 1. calibrate ──► p_opt(N, machine) ──► 1b. budget --fill ──► 2. run ──► 3. report
              (no QPU)      (QPU, ~9 s)      calibration/p_opt.csv (no QPU: sets reads)   (QPU, the rest)  results/sk_qpu_results.csv
```

Each step is resumable: rerunning a command skips what is already saved. Every QPU command has a hard cap, `--budget-seconds`.

## What is being calibrated

Each agent i is a chain of qubits on the chip (a "hyper-qubit"). The physical problem has two kinds of couplings:

| couplers | value | depends on p? |
|---|---|---|
| inside chain i (keep its qubits equal) | −c, with **c = p · J_rms · √(N−1)** | **yes** |
| between chains i and j (the SK problem) | W_ij split over the couplers linking the two chains | no |

So p does not multiply the logical couplings. It sets how strong the chains are **compared with the field an agent feels from the others**: J_rms · √(N−1) is the typical size of that field. The QPU then divides everything by roughly c to fit its range [−1, 1].
* p too small: chains break and the read is not a valid assignment of the N agents (it is repaired by majority vote).
* p too large: the SK couplings become tiny compared with the QPU's analogue noise.

The best p depends on N and on the embedding, so on the machine. Calibration finds, once per machine and N and shared by every problem class, **p_opt(N, machine) = the smallest p of the grid 0.75, 1, 1.414, 2, 3, 4, 6 with no broken chain in any of 200 reads**. It climbs the grid from the smallest p and stops as soon as one p meets the rule. The derivation of why p ≈ 3–4 is expected at large N is in [`../../prefactor_and_shimming.pdf`](../../prefactor_and_shimming.pdf).

## The commands (PowerShell; in bash write `F=results/sk`)

All 4 problem classes, every N, and the whole budget. `--total` is the QPU time left in your Leap account **before** you start (check it in the Leap dashboard; ≈178 s). Pass the same value at every step: the time the campaign has already spent is read from its own results and subtracted.

```powershell
$F = "results\sk"

# setup: the list of N, all 4 problem classes, 5 instances each (no QPU)
python main_simulation.py init --preset full --out $F --Ns 8:160:8 --n-instances 5

# 0. predicted cost (no QPU)
python main_simulation.py budget --out $F --total 178

# 1. calibration: one p_opt for every (N, machine), shared by all problem classes
python main_simulation.py calibrate --out $F --budget-seconds 15
#    -> results\sk\calibration\p_opt.csv and p_opt.png; the table is stored in config.json

# 1b. size the run to spend everything that is left (no QPU): sets num_reads in config.json
python main_simulation.py budget --out $F --total 178 --fill

# 2. the SK problems, every (class, N, machine) with its own p_opt
python main_simulation.py run --out $F --check                    # estimate only
python main_simulation.py run --out $F --total 178                # never spends more than 178 - what the campaign already spent

# 3. results (no QPU; also printed at the end of step 2)
python main_simulation.py report --out $F
#    -> results\sk\results\sk_qpu_results.csv
```

* `--Ns 8:160:8` means 8, 16, …, 160. Zephyr embeds the full SK problem up to N = 104, so larger N run on Pegasus only (up to 177). Both steps skip the N a machine cannot embed.
* **One calibration per (topology, N), shared by every class.** The embedding (the chains) depends only on the machine and N, so p_opt is measured once, on the pure SK class (`calibrate --problem KEY` to choose another one), and `run` uses it for every class. It refuses to start if a (N, machine) is missing. Caveat: the chains feel the fields of the problem, and the ferromagnetic class (J0 = 3) has larger ones. Measured largest field in units of J_rms·√(N−1): 3.69 for SK and 5.23 for J0 = 3 at N = 104. So its chains can break somewhat more often at the shared p. `report` and `compare` show the broken-chain fraction per class, so this is visible, not hidden.
* **`budget --fill`** computes the reads per call that make step 2 spend: total − already spent − calibration still to do − 3 s margin (`--margin`). One call cannot exceed the QPU's 1 s limit, so the reads are capped at that, and the cap is printed. Run it after `calibrate`; before calibration, it reserves the worst-case calibration cost.
* If a step stops at its budget, rerun the same command: it continues where it stopped. N are done in increasing order, all classes and both machines at each N.
* To run only some classes: `init ... --regimes sg_satisficing` (pure SK) or choose the calibration class with `calibrate --problem eps0_J00`.

## What the output contains

`results\sk\results\sk_qpu_results.csv`, one row per (machine, problem, N, instance):

| column | meaning |
|---|---|
| `p_used`, `chain_strength` | p_opt of that (N, machine) and the resulting c |
| `best_configuration` | the best read: `+`/`-` per agent, agent 0 written as `+` (S and −S are the same state) |
| `energy_H`, `reward_R` | its energy H = −Σ W_ij S_i S_j and reward R_N = −2H/N |
| `reward_R_polished` | the same after a greedy single-flip descent of every read |
| `reference_R`, `reference_source` | the optimum to compare with: `exact` (enumeration, N ≤ 24), or the best classical solution if the `classical` stage has run; otherwise empty |
| `reached_reference`, `gap_R` | whether the QPU's best (raw or polished) equals the reference, and by how much it misses it |
| `frac_reads_no_broken_chain`, `qpu_access_s` | chain quality and the QPU time charged for the call |

For N > 24 the reference needs the classical stage (`python main_simulation.py classical --out $F`, slow at large N; see README_LOCAL.md). Without it the QPU's configuration is reported, but not compared.

## Step 4: compare (no QPU) — the planner question

```powershell
python main_simulation.py classical --out $F            # CPU only: classical solvers, the game, references
python main_simulation.py compare   --out $F            # -> results\sk\compare\compare.csv, compare.png, reference_quality.png
```

The question: how well does the QPU act as the **central planner** (it picks every agent's choice so as to maximise the total reward)? It is compared with **classical planners** (greedy, SA, PT, τ-EO, SDP+GW, dimod SA/tabu) and with **no planner** (the game: agents learning on their own). Problem classes go from easy to hard: `eps0_J03` (ferromagnetic) → `eps0_J01` (boundary) → `eps0_J00`, `eps1_J00` (spin glass).

**Reference.** For N ≤ n_exact (24) it is the exact optimum. Above that it is the **best reward found by any method**, which is not certified (grey band in the figure). `reference_quality.png` checks how plausible it is: for pure SK the mean ground-state reward should follow R(N) ≈ −2·(−0.7633 + 0.7·N^(−2/3)). That is the Parisi energy plus its finite-size correction.

**Metrics.** A *sample* is one QPU read, one run of a classical solver, or one run of the game. The residual is ρ = (R_ref − R)/|R_ref|.

| metric | meaning | useful when |
|---|---|---|
| `rho_best_median` | residual of the best sample, median over instances | every N |
| `rho_typical_median` | residual of the median sample | every N: is a typical answer close? |
| `P_rho_le_0` | fraction of samples that reach the reference exactly | small N; it decays exponentially with N for every heuristic, so above n_exact it is ≈ 0 and measured against an uncertified reference |
| `P_rho_le_0.01` (also 0.02, 0.05) | fraction of samples within 1 % of the reference: **satisficing success** | every N: the primary metric |
| `TTT99_0.01_s` | time to reach ρ ≤ 1 % once with 99 % probability = t·ln 0.01/ln(1−P) | time comparison. QPU t = anneal + readout + delay per read; `*_incl_programming` also counts the programming time; classical t = wall time of one run. Polished QPU reads have no TTT (polishing runs on the CPU) |

 (≈178 s, all 4 classes)

* **Calibration.** Zephyr N = 8…104 is 13 N; Pegasus N = 8…160 is 20 N, so 33 (N, machine) pairs, shared by the 4 classes.
  * One call: ≈ 15.8 ms + 200 reads × 0.2 ms ≈ 55.8 ms.
  * Upper bound, all 7 p: 33 × 7 × 0.0558 ≈ **12.9 s**.
  * Expected, since the climb stops at p_opt ≈ 3 (the 5th value): 33 × 5 × 0.0558 ≈ **9.2 s**.
* **Run, filled.** Left: 178 − 9.2 − 3 (margin) ≈ 165.8 s.
  * Calls: 4 classes × 33 pairs × 5 instances = 660.
  * Fixed part: 660 × 15.8 ms ≈ 10.4 s.
  * Each read costs ≈ 0.2 ms per call, so 660 × 0.2 ms ≈ 0.132 s per read of every call.
  * Reads per call: (165.8 − 10.4) / 0.132 ≈ **≈ 1180** (the readout is longer for large N, so `--fill` will find somewhat fewer).
* `budget` and `budget --fill` print the exact figures from each machine's timing model.

## Optional extras (not needed for the pipeline)
* `shim` (flux-bias shimming) and `--extended-j-range`: see README_LOCAL.md and prefactor_and_shimming.pdf.
* `validate`, `ground`, `chains`, `analyze`: checks and figures on the same directory, no QPU.

## Before spending QPU time: is the problem feasible at all? (`feasibility`, no QPU)

```powershell
python main_simulation.py feasibility --topology zephyr  --N 16 32 64 104
python main_simulation.py feasibility --topology pegasus --N 16 64 104 160 --J0 3 --plot feas.png
python main_simulation.py feasibility --topology zephyr  --N 64 104 --extended-j-range --sigma 0.01
```
For each N it prints YES / MARGINAL / NO and why. It combines:
- the chain length of the clique embedding;
- the chain coupling the problem's fields require (p_low);
- the compression into the hardware range;
- the control error σ;
- the effective temperature of the machine.

It also prints the σ and β_eff the problem **would need** for a YES. After a run, `python -m dwave_utils noise-fit --out $F` measures both on each machine from the small-N reads, and `feasibility ... --from-fit $F` uses the measured values. σ (default 0.02) and β_eff (default 5) are assumed orders of magnitude, not measurements. YES is a necessary condition for a faithful, cold sampler; it does not promise that the QPU beats classical solvers. The formulas are in the docstring of `dwave_utils/feasibility.py`.
