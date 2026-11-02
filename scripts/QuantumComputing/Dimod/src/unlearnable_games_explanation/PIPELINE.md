# The QPU pipeline, step by step

**Input:** a list of N (number of agents) and the machines (Zephyr = Advantage2, Pegasus = Advantage).
**Output:** for every (machine, problem class, N, instance): the best configuration found (ground-state candidate), its energy, its reward R_N, and whether it equals the reference optimum.

```
list of N ──► 0. budget ──► 1. calibrate ──► p_opt(class, N, machine) ──► 1b. budget --fill ──► 2. run ──► 3. report
              (no QPU)      (QPU, ~37 s)     calibration/p_opt.csv        (no QPU: sets reads)   (QPU, the rest)  results/sk_qpu_results.csv
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

The best p depends on N and on the embedding, so on the machine. Calibration finds, for every problem class too, **p_opt(N, machine) = the smallest p of the grid 0.75, 1, 1.414, 2, 3, 4, 6 with no broken chain in any of 200 reads**. It climbs the grid from the smallest p and stops as soon as one p meets the rule. The derivation of why p ≈ 3–4 is expected at large N is in [`../../prefactor_and_shimming.pdf`](../../prefactor_and_shimming.pdf).

## The commands (PowerShell; in bash write `F=results/sk`)

All 4 problem classes, every N, and the whole budget. `--total` is the QPU time left in your Leap account **before** you start (check it in the Leap dashboard; ≈178 s). Pass the same value at every step: the time the campaign has already spent is read from its own results and subtracted.

```powershell
$F = "results\sk"

# setup: the list of N, all 4 problem classes, 5 instances each (no QPU)
python main_simulation.py init --preset full --out $F --Ns 8:160:8 --n-instances 5

# 0. predicted cost (no QPU)
python main_simulation.py budget --out $F --total 178

# 1. calibration: p_opt for every (problem class, N, machine)
python main_simulation.py calibrate --out $F --budget-seconds 55
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
* **Why calibrate per problem class:** p_opt depends on the field the chains feel. The ferromagnetic class (J0 = 3) has fields about 3× larger in its ordered ground state, so it needs a larger p than the pure SK class. `run` uses, for every call, the p_opt of its own (class, N, machine). It refuses to start if one is missing.
* **`budget --fill`** computes the reads per call that make step 2 spend: total − already spent − calibration still to do − 3 s margin (`--margin`). One call cannot exceed the QPU's 1 s limit, so the reads are capped at that, and the cap is printed. Run it after `calibrate`; before calibration, it reserves the worst-case calibration cost.
* If a step stops at its budget, rerun the same command: it continues where it stopped. N are done in increasing order, all classes and both machines at each N.
* To run only some classes: `init ... --regimes sg_satisficing` (pure SK) or `calibrate --problem eps0_J00`.

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

## Budget, step by step (≈178 s, all 4 classes)

* **Calibration.** Zephyr N = 8…104 is 13 N; Pegasus N = 8…160 is 20 N, so 33 (N, machine) pairs. With 4 classes that is 132 calibrations.
  * One call: ≈ 15.8 ms + 200 reads × 0.2 ms ≈ 55.8 ms.
  * Upper bound, all 7 p: 132 × 7 × 0.0558 ≈ **51.6 s**.
  * Expected, since the climb stops at p_opt ≈ 3 (the 5th value): 132 × 5 × 0.0558 ≈ **37 s**.
* **Run, filled.** Left: 178 − 37 − 3 (margin) ≈ 138 s.
  * Calls: 4 classes × 33 pairs × 5 instances = 660.
  * Fixed part: 660 × 15.8 ms ≈ 10.4 s.
  * Each read costs ≈ 0.2 ms per call, so 660 × 0.2 ms ≈ 0.132 s per read of every call.
  * Reads per call: (138 − 10.4) / 0.132 ≈ **≈ 960** (the readout is longer for large N, so `--fill` will find somewhat fewer).
* `budget` and `budget --fill` print the exact figures from each machine's timing model.

## Optional extras (not needed for the pipeline)
* `shim` (flux-bias shimming) and `--extended-j-range`: see README_LOCAL.md and prefactor_and_shimming.pdf.
* `validate`, `ground`, `chains`, `analyze`: checks and figures on the same directory, no QPU.
