# The QPU pipeline, step by step

**Input:** a list of N (number of agents) and the machines (Zephyr = Advantage2, Pegasus = Advantage).
**Output:** for every (machine, N, instance): the best configuration found (ground-state candidate), its energy, its reward R_N, and whether it equals the reference optimum.

```
list of N ──► 0. budget ──► 1. calibrate ──► p_opt(N, machine) ──► 2. run ──► 3. report
              (no QPU)      (QPU, ~5–13 s)     calibration/p_opt.csv   (QPU)     results/sk_qpu_results.csv
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

The best p depends on N and on the embedding, so on the machine. Calibration finds **p_opt(N, machine) = the smallest p of the grid 0.75, 1, 1.414, 2, 3, 4, 6 with no broken chain in any of 200 reads**. It climbs the grid from the smallest p and stops as soon as one p meets the rule. The derivation of why p ≈ 3–4 is expected at large N is in [`../../prefactor_and_shimming.pdf`](../../prefactor_and_shimming.pdf).

## The commands (PowerShell; in bash write `F=results/sk`)

```powershell
$F = "results\sk"

# setup: the list of N and how much to run per N (no QPU)
python main_simulation.py init --preset full --out $F --Ns 8:160:8 --regimes sg_satisficing --n-instances 5 --num-reads 1000

# 0. how much QPU time steps 1 and 2 will cost (no QPU; prints whether it fits)
python main_simulation.py budget --out $F --total 178

# 1. calibration: p_opt for every (N, machine)
python main_simulation.py calibrate --out $F --budget-seconds 15
#    -> results\sk\calibration\p_opt.csv and p_opt.png; the table is stored in config.json

# 2. the SK problems, every N with its own p_opt (refuses to start if some (N, machine) is not calibrated)
python main_simulation.py run --out $F --check                    # estimate only
python main_simulation.py run --out $F --budget-seconds 160

# 3. results (no QPU; also printed at the end of step 2)
python main_simulation.py report --out $F
#    -> results\sk\results\sk_qpu_results.csv
```

* `--Ns 8:160:8` means 8, 16, …, 160. Zephyr embeds the full SK problem up to N = 104, so larger N run on Pegasus only (up to 177). Both steps skip the N a machine cannot embed.
* `--regimes sg_satisficing` keeps only the pure SK problem (ε = 0, J0 = 0). Without it all 4 problem classes run, and step 2 costs about 4× more.
* If step 2 stops at its budget, rerun the same command: it continues from the next N. N are done in increasing order, both machines at each N.

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

## Budget (≈178 s left, to use by 4 November)

* Step 1: 13 N on Zephyr (8…104) + 20 N on Pegasus (8…160) = 33 (N, machine) pairs. At most 33 × 7 = 231 calls of ~55 ms, so **≤ 13 s**. Usually about half, because the climb stops at p_opt.
* Step 2: the cost per (N, machine) is about 15.8 ms + reads × ~0.2 ms per call, times the number of instances. With 1 class, 5 instances and 1000 reads, all N on both machines take **≈ 40 s**. With all 4 classes it is ≈ 157 s.
* `budget` prints the exact prediction from each solver's timing model. Change `--n-instances` / `--num-reads` in `init --force` until it fits.

## Optional extras (not needed for the pipeline)
* `shim` (flux-bias shimming) and `--extended-j-range`: see README_LOCAL.md and prefactor_and_shimming.pdf.
* `validate`, `ground`, `chains`, `analyze`: checks and figures on the same directory, no QPU.
