# What this project taught us about quantum annealers on all-to-all problems

These lessons come from using D-Wave QPUs (Zephyr / Advantage2 and Pegasus / Advantage) as the central planner of the SK game. The planner's problem is a fully connected Ising model, H = −Σ_{i<j} W_ij S_i S_j. The lessons hold for any dense problem on a sparse annealer.

Some numbers below are **measured**: embeddings, fields, run times, the first real calls. Others rest on **assumed** machine parameters: the control error σ ≈ 0.02 and the effective inverse temperature β_eff ≈ 5. `python -m dwave_utils noise-fit` replaces the assumed values with values measured from the QPU's own reads.

## 1. A dense problem becomes long chains

The chip is sparse: each qubit couples to 15 (Pegasus) or 20 (Zephyr) others. To couple every agent to every other, each agent i becomes a **chain** of L qubits forced to agree. This is the clique embedding.

| N | 16 | 64 | 104 |
|---|---|---|---|
| chain length L (Zephyr) | 3 | 6 | 8 |
| qubits used ≈ N·L | 48 | 384 | 832 |

The largest fully connected problem is **N = 104 on Zephyr** (Advantage2_system1 reports 105) and **N = 177 on Pegasus** (Advantage_system4), out of thousands of qubits. Qubits grow like N²/12, so the chip's size is not the limit: its connectivity is.

## 2. A chain must beat a field, not a coupling

In a dense problem each agent feels the sum of N−1 couplings, its local field h_i = Σ_j W_ij S_j. For SK, W_ij ~ N(0, 1/N), so a single coupling is about 1/√N while the field is O(1). A chain of strength c holds only if

  **c ≥ max_i |h_i|** at the low-energy states the anneal reaches.

In units of the typical field u = J_rms·√(N−1), the required prefactor p_low = max_i|h_i|/u is:

| class | N = 16 | N = 64 | N = 104 | N = 160 |
|---|---|---|---|---|
| spin glass (`eps0_J00`, `eps1_J00`, `eps0_J01`) | 2.6–2.8 | 3.5–3.6 | 3.7–3.8 | 3.8–3.9 |
| ferromagnetic (`eps0_J03`) | 3.6 | 4.9 | 5.2 | 5.5 |

It grows slowly with N, like the largest of N Gaussians (√(2 ln N)). The worst case over all states, max_i Σ_j |W_ij| ≈ √(2/π)·√(N−1), is far larger and irrelevant: such states are never reached.

**Lesson:** "chain strength ≈ 3 × the largest coupling" is a rule of thumb. What actually matters is chain strength relative to the field, and it depends on the problem class. The ferromagnetic class needs about 40 % more.

## 3. The chip has one fixed full scale, so only ratios can be chosen

Every coupler accepts J ∈ [−1, 1]. Chain couplers may go to −2 with the extended range, since they are always ferromagnetic. The QPU divides the whole problem by its largest coupling, which is the chain coupling c. So:

- chain couplers end at −1, the hardware maximum: **raising c cannot make chains stronger in absolute terms**;
- every SK coupling shrinks to W_ij/c, with rms

  **1/(p·√(N−1))** of the full range. At N = 104 and p = 3.6: 1/(3.6 × 10.15) ≈ **0.027**, i.e. 2.7 % of the full range.

## 4. Two errors grow as the problem shrinks

**Control error (wrong landscape).** Each coupler is realised with an absolute error σ ≈ 0.01–0.03 (calibration, crosstalk). Relative to an SK coupling it is

  **ε = σ·√|E|·p·√(N−1)**   (|E| ≈ 1.1–1.2 couplers between two chains)

That is ≈ 0.31 at N = 16, ≈ 0.59 at N = 64 and ≈ 0.81 at N = 104 for σ = 0.02. The chip then holds a *different instance*. SK ground states are sensitive to small coupling changes ("bond chaos"), so lowering the temperature cannot fix this: at β → ∞ the QPU would return the ground state of the wrong problem.

**Temperature (hot sampling).** The chip samples roughly exp(−β_eff·H_chip), with β_eff fixed in hardware units. In field units the temperature is

  **T = p / (r·β_eff)**   (r = 2 with the extended range, otherwise 1)

So stronger chains make the problem *hotter*. The SK spin glass freezes at T_c = 1 in these units, and with p ≈ 3.6 and β_eff ≈ 5, T ≈ 0.7 T_c.

## 5. The window, and why calibration cannot remove it

p is squeezed from both sides:

  **p_low (chains hold)  ≤  p  ≤  min( 1/(σ·√|E|·√(N−1)) ,  r·β_eff·T_c·t_ok )**   (noise, temperature)

The lower bound rises slowly, like √(ln N). The noise bound falls like 1/√N. They cross around N ≈ 1/(p_low·σ)², which is ≈ 200 for σ = 0.02 and ≈ 90 for σ = 0.03. Past that point no choice of p is good: either chains break or the problem drowns.

This is why the calibration picks the **smallest p with no broken chain**. Any larger p knowingly compresses the problem further. Calibration can find the best point of the window; it cannot widen it.

The feasibility model (`python -m dwave_utils feasibility`) evaluates exactly this for a topology, N and problem. With σ = 0.02 and β_eff = 5, all-to-all SK is **MARGINAL** on Zephyr at every N. To be a YES it would need σ ≤ 0.0074 and β_eff ≥ 12 at N = 104. With the ferromagnetic class on Pegasus it is a **NO** from N ≈ 100.

## 6. What does widen the window, and by how much

| remedy | effect | limit |
|---|---|---|
| extended J range (chains to −2) | SK couplings ×2: ε and T halve | per-qubit coupling sums; extended couplers may be less precise |
| flux-bias shimming | removes the *systematic* part of the errors (biases ⟨s_q⟩ ≠ 0 with h = 0) | random errors stay; costs ~1.5–2 s per machine |
| accept a few broken chains (`--threshold`) | smaller p: less compression and colder | the repaired reads are biased |
| polishing the reads (greedy descent) | recovers the bottom of the valley the QPU found | classical work; not counted in the QPU time |
| a sparser formulation of the problem | short or no chains: the whole issue disappears | changes the model |

What cannot change: a dense problem on a sparse chip means long chains, and long chains, a fixed full scale and a fixed absolute noise mean a problem compressed into that noise.

## 7. How to measure success honestly

- **Exact success probability is the wrong metric at large N.** For every heuristic, the QPU included, the probability per sample of the exact ground state decays like e^{−aN}. With R reads nothing below 1/R can be measured. Above the size where exact enumeration works (N ≈ 24), the "optimum" is the best anyone found, which is circular and uncertified.
- **Use closeness.** Report the residual ρ = (R_ref − R)/|R_ref| of the best and of the typical sample, and the **satisficing success** P(ρ ≤ 1 %), which stays meaningful at every N.
- **Check the reference against theory.** The mean SK ground state follows R(N) ≈ −2(−0.7633 + 0.7·N^{−2/3}) (Parisi energy plus its finite-size correction). A best-known reference well below that curve is not the ground state.
- **Time comparisons need a definition.** QPU time per read (anneal + readout + delay ≈ 0.1–0.2 ms) is not the same as the cost of a call (programming ≈ 16 ms) or the wall time including the network. Report both, and give polished reads no QPU time.
- **Compare against no planner too.** The question "does the QPU plan better than agents learning on their own (the game)?" can have a different answer from "does it beat simulated annealing?".

## 8. Measure the machine, don't assume it

σ and β_eff decide the verdict, and they are properties of the specific chip on the day. `noise-fit` measures them from the small-N reads, where all 2^(N−1) states can be enumerated. Its model has fixed per-call coupler errors plus Boltzmann sampling, with the errors integrated out by a Laplace approximation; validated on synthetic data, it recovers both parameters. Feed the result into `feasibility --from-fit` to turn the forecast into a measurement.

## 9. Practical lessons from running it

- **Budget.** A call costs ≈ 15.8 ms of programming plus the reads (e.g. 157 ms for an N = 8 call on Zephyr, 175 ms for N = 16). The solver's timing model over-estimated by ≈ 10 % (173 ms predicted vs 157 ms charged), so a guard with a safety factor is cheap insurance. One calibration per (machine, N), shared by all problem classes, costs ≈ 9 s for 33 pairs. The rest of the budget goes to reads.
- **The chip changes.** The working graph of a QPU changed between two days. A cached embedding then contains dead qubits (`DisconnectedChainError`), so embeddings must be re-validated against today's graph before every run.
- **Record everything.** For each call: the problem id, the solver snapshot and a fingerprint of its graph, the embedding, the chain strength, the exact physical problem sent and every read. With that, every result can be re-derived and validated later.
- **A mock QPU is not a QPU.** The pipeline was tested on D-Wave's mock sampler, which is classical simulated annealing with little effort. Its curves look "bad" and say nothing about the hardware.
- **First real data:** see §11.
- **Windows and synced folders:** see §12.

## 10. The decision rule

Before running a dense problem on an annealer:

1. **Fit:** N ≤ the largest clique of the machine (Zephyr 104, Pegasus 177)?
2. **Chains:** what p_low do the problem's fields need? (`feasibility`)
3. **Noise:** is ε = σ·√|E|·p_low·√(N−1)/r well below 1?
4. **Temperature:** is p_low/(r·β_eff) well below the problem's T_c?
5. **Measure** σ and β_eff on the machine (`noise-fit`), and repeat 3–4.
6. If all pass, run and compare **closeness and time** against classical solvers; if not, reformulate the problem, use the extended range or shimming, or use a classical solver.

## 11. The first real results: what the report says (N = 8, 16, 24)

The first run on the real machines covered all four problem classes, 5 instances each: N = 8 and 16 on both QPUs, and N = 24 on Zephyr. It then stopped on the file lock of §12. `report` gives one row per (machine, class, N, instance):

| column | meaning |
|---|---|
| `p` | the calibrated chain-strength prefactor used for that (machine, N) |
| `H` | energy of the best read, H = −Σ W_ij S_i S_j |
| `R_N` | its reward, R_N = −2H/N |
| `polished` | the best read after greedy single-flip descent |
| `ref R_N` | the reference: for N ≤ 24 the **exact** optimum, by enumeration of all 2^(N−1) configurations |
| `reached` | does the QPU's best (raw or polished) equal the reference? |

**What it shows.**

1. **Every one of the 100 rows is `True`.** Both QPUs found the exact ground state of every instance, in every class, at N = 8, 16 and 24. For N = 24 that is one configuration out of 2^23 ≈ 8.4 million. Random guessing with a few thousand reads would essentially never find it, so the QPU is concentrating its reads at low energy.
2. **Raw = polished = reference.** The best read is the optimum without any classical help: greedy polishing had nothing left to fix.
3. **Both machines agree** on every N = 8 and 16 instance: same instances, same optimum.
4. **The calibrated p is smaller than the theory bound.**
   - Zephyr needed p = 0.75 (N = 8) and 1 (N = 16, 24); Pegasus 1.414 (N = 8) and 2 (N = 16).
   - The bound of §2 gives p_low ≈ 2.6–2.9 at N = 16, so it is **conservative**. It assumes the full field pushes against a chain at the end of the anneal. In reality chains freeze earlier, while the transverse field is still strong and the effective fields are weaker, and short chains (L = 3) are hard to break.
   - **Pegasus needs about twice the p of Zephyr.** Its qubits have fewer couplers (15 vs 20), so its chains are longer for the same N, and longer chains break more easily.
5. **The rewards make physical sense.**
   - Ferromagnetic class `eps0_J03`: R ≈ 2.5–3.4. The agents align, and R ≈ J0 = 3 plus fluctuations.
   - Boundary class `eps0_J01`: R ≈ 1.2–1.9.
   - Pure SK `eps0_J00`: R ≈ 0.8–1.5. At N = 24 the mean of the 5 instances is 1.29, close to the Parisi finite-size value 1.36 given 5 instances.
   - Non-reciprocal class `eps1_J00`: R ≈ 0.5–1.0, about half of SK, because the planner only sees the symmetric half of J.

**What it does not show.** "The best of all reads is the optimum" says nothing about **how often** a read is optimal, or how good a typical read is. Those are `compare`'s P(ρ = 0), P(ρ ≤ 1 %) and the typical residual. Nor does it say anything about large N: up to N = 24 the feasibility model gives ε ≈ 0.3–0.4, so the QPU is still in its comfortable regime. The interesting part of the experiment is the larger N, where §5 predicts the window closes.

## 12. A Windows lesson: synced folders and cache files

The run stopped with

```
PermissionError: [WinError 5] Access denied: '...\.venv\data\busclique_7\clique\.lru~' -> '...\.lru'
```

**What happened.**
- To find clique embeddings fast, minorminer keeps a **cache on disk**, inside the virtual environment (`.venv\data\busclique_7`).
- To update it safely, it writes a temporary file `.lru~` and then renames it to `.lru`.
- On Windows a file cannot be renamed while another program holds it open. The project, `.venv` included, lives in **OneDrive**, which opens files to sync them; an antivirus can do the same.
- So the rename was refused, Python raised `PermissionError`, and the run stopped at the start of the next (machine, N). No QPU time was lost: the lock happens before any call, and every finished call was already saved.

**Why it was not under our control.** minorminer has no option to move its cache, and the lock is transient: it depends on when OneDrive looks at the file.

**What the code now does** (`skgame/optim/embedding.py`):
1. **Fewer disk accesses:** the largest clique of each machine is computed once and remembered in memory. Before, it was read from disk at every N.
2. **Retries:** on a locked file it waits 0.5, 1, 2 and 4 s and tries again. A sync usually finishes in that time.
3. **Fallback:** if the file stays locked, it computes **without the disk cache**. The result is the same, only slower.
   - Uncached cliques of 8 or fewer variables crash minorminer 0.2.22 (`std::bad_alloc`), so for small N it embeds a K_12 and keeps the first N chains. That is still a valid embedding of K_N.
   - A test simulates both locks and checks the results.

**General lesson.** Long, stateful runs should not live in a synced folder (OneDrive, Dropbox, Google Drive): the sync client competes for files, caches and git's own `.git` files. Keep the repository and the virtual environment in a local folder such as `C:\Users\<you>\Projects`, or at least pause syncing during a run. Code that writes caches should treat a locked file as a temporary condition: retry, then fall back.

Documents: [`prefactor_and_shimming.pdf`](prefactor_and_shimming.pdf) (derivations), [`dwave_in_the_code.pdf`](dwave_in_the_code.pdf) (code), [`src/unlearnable_games_explanation/PIPELINE.md`](src/unlearnable_games_explanation/PIPELINE.md) (commands).
