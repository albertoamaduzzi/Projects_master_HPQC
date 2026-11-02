# What a campaign saves, and how to reconstruct everything a posteriori

Everything below lives in the campaign directory (`--out`, e.g. `results\full`). Together it is enough to rebuild every input given to the QPU and to re-derive every number from scratch: the instances, the exact physical problem, the scaling, the embedding, the reads. `python main_simulation.py validate --out DIR` does it automatically.

## Files

| file | content |
|---|---|
| `config.json` | every campaign setting: N grid, regimes, instances, reads, annealing time, chain-strength prefactor, chain-break method, back-ends, seed |
| `instances/<class>/N0024.npz` | **the SK game inputs**: `J` (instances × N × N, the full coupling matrices, asymmetric for ε > 0), `eps`, `J0`, `N`. The planner's problem uses `W = (J + J.T)/2` with zero diagonal |
| `classical/<class>/N0024.json` | reference per instance: `ref` (R_N), `ref_S` (configuration), `ref_source` (`exact` for N ≤ 24, else the solver that found it), `ref_exact`; every classical solver's rewards and times; SDP upper bounds |
| `game/<regime>/N0024.json` | the game's dynamics: best / mean / final reward and period (phase) for every instance and start |
| `embeddings/<solver>/clique/N0024.json` | the cached embedding on that QPU's graph (re-validated before each use; a replaced one is kept as `*.stale.json`) |
| `anneal/<backend>/<class>/N0024_i0.json` | **one QPU (or simulator) call**: see below |
| `shims/<backend>/fbos.json`, `history.json`, `shim_history.png` | flux-bias offsets learned by `shim` (qubit → Φ0) and, per iteration, magnetisations, offsets, chain breaks, QPU time, chip fingerprint |
| `summary/*.csv`, `*.png` | analysis: `summary.csv`, `backend_comparison.csv`, `chain_breaks.csv`, `ground_states.csv` |

## One annealer call: `anneal/<backend>/<class>/N…_i….json`

| field | meaning |
|---|---|
| `key`, `eps`, `J0`, `N`, `instance`, `backend` | which problem: instance `instance` of `instances/<key>/N….npz` |
| `rewards`, `counts` | every distinct configuration's reward R_N and how many reads gave it (best first) |
| `best_reward`, `best_S` | the best read and its configuration (`+`/`-`, agent 0 first) |
| `polished_*` | the same after greedy single-flip descent of every read |
| `reads.S`, `reads.reward`, `reads.polished_reward`, `reads.n_broken`, `reads.count` | **every distinct (configuration, number of broken chains)** with its reward, polished reward and number of reads |
| `info.problem_id` | D-Wave's id of the problem: findable in the Leap dashboard together with `info.sample_kwargs.label` |
| `info.solver`, `info.solver_properties` | QPU name; `chip_id`, `topology` (type, shape), `h_range`, `j_range`, `extended_j_range`, working qubits/couplers, `graph_sha1` (fingerprint of the working graph at that moment) |
| `info.settings`, `info.sample_kwargs` | annealing settings and the exact keyword arguments sent (`num_reads`, `annealing_time`, `label`) |
| `info.embedding_used` | **the embedding of that call**: logical spin → list of qubits |
| `info.chain_lengths`, `info.embedding` | chain lengths; qubits used, mean / max chain length |
| `info.chain_strength` | c actually used = prefactor × J_rms × sqrt(N−1) |
| `info.programmed` | the physical problem: `max_abs_J_before_scaling`, `auto_scale_divisor` (the QPU divides every coupler by it), `n_qubits`, `n_couplers` |
| `info.programmed.scaling` | `mode` (`auto_scale`, or `extended_j_range` with its `divisor`: the code then scales the problem itself and sends `auto_scale=False`) |
| `info.flux_biases` | flux-bias offsets applied (from `shim`): source file, number of non-zero offsets, rms, max |
| `info.max_abs_logical_coupling` | max |W_ij| of the instance |
| `info.broken_chains_hist`, `info.chain_break_count`, `info.chain_break_fraction_mean`, `info.discarded_reads` | reads with 0, 1, 2… broken chains; how often each spin's chain broke; D-Wave's mean fraction |
| `info.physical_problem_file` | `N…_i…_physical.npz`: qubits `h_q` / fields `h`, couplers `u`, `v` / values `J`, `offset`: the exact physical Ising problem, before the QPU's auto-scaling |
| `info.timing_us` | D-Wave's timing: `qpu_access_time` (what is charged), programming, sampling, anneal / readout / delay per read |
| `info.estimated_access_s` | the prediction made before the call (budget limit) |

## Rebuild a call by hand

```python
import json, numpy as np, dimod
from dwave.embedding import EmbeddedStructure
from skgame.optim.problem import PlannerProblem

out, key, N, b, backend = "results/full", "eps0_J00", 64, 0, "qpu_zephyr_clique"
J = np.load(f"{out}/instances/{key}/N{N:04d}.npz")["J"][b]          # the SK game's coupling matrix
P = PlannerProblem(J)                                                 # W = (J + J^T)/2, zero diagonal
r = json.load(open(f"{out}/anneal/{backend}/{key}/N{N:04d}_i{b}.json"))
info = r["info"]

# 1. the logical problem (Ising; P.to_qubo() gives the equivalent QUBO)
bqm = P.to_bqm()                                                      # h = 0, J_ij = -W_ij
# 2. the physical problem actually programmed (before auto_scale), saved exactly
ph = np.load(f"{out}/anneal/{backend}/{key}/N{N:04d}_i{b}_physical.npz")
physical = dimod.BinaryQuadraticModel(dict(zip(ph["h_q"].tolist(), ph["h"].tolist())),
                                      dict(zip(zip(ph["u"].tolist(), ph["v"].tolist()), ph["J"].tolist())),
                                      float(ph["offset"]), "SPIN")
#    ... and it is exactly the embedding of the instance with the saved embedding and chain strength:
emb = {int(k): v for k, v in info["embedding_used"].items()}
rebuilt = EmbeddedStructure(list(physical.quadratic), emb).embed_bqm(bqm, chain_strength=info["chain_strength"],
                                                                     smear_vartype=dimod.SPIN)
print(rebuilt.is_almost_equal(physical))                              # True
# 3. the scaling the QPU applied
print("divide by", info["programmed"]["auto_scale_divisor"])
# 4. the solutions
S = np.array([[1 if ch == "+" else -1 for ch in s] for s in r["reads"]["S"]])
print(np.allclose(P.reward(S), r["reads"]["reward"]))                # every read re-checked
print("energies H:", P.energy(S)[:5], "broken chains:", r["reads"]["n_broken"][:5])
```
The physical problem depends on the chip's working graph that day: how many couplers link two chains, and therefore how W_ij is split. That is why it is saved, not recomputed. `info.solver_properties.graph_sha1` identifies the graph.

## Check everything automatically

```powershell
python main_simulation.py validate --out results\full
```
It recomputes, from the stored inputs:
* the reward of every stored configuration (reads, best, polished) and of the classical reference;
* the exact optimum for N ≤ 24;
* that read counts add up, and that the broken-chain counts agree with D-Wave's `chain_break_fraction`;
* the chain strength from the formula;
* that every `embedding_used` is a valid embedding of K_N on the solver's topology, and that the chain lengths match.
