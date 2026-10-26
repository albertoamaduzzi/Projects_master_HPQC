# unlearnable_games_explanation

Reproduction toolkit for J. Garnier-Brun, M. Benzaquen, J.-P. Bouchaud, *Unlearnable Games and "Satisficing" Decisions: A Simple Model for a Complex World*, arXiv:2312.12252 — plus a benchmark of optimisers for the planner's problem, ready for a quantum-annealing back-end.

```
docs/
  01_mathematical_structure.md   model, limits, welfare, NMFE, complexity, DMFT, RS instability — all derivations expanded
  02_objects.md                  list of mathematical objects            (generated from skgame/mathcat.py)
  03_computations.md             list of computations + parameters       (generated)
  04_visualizations.md           list of figures, reading order, build order (generated)
  05_dependencies.md             argument DAG (claims ← results) + class category (generated block)
skgame/                          Python package (NumPy/SciPy/Numba numerics, SymPy proofs)
  model.py                       InteractionEnsemble, GameParams, SKGame, recorders
  observables.py                 R_N, e_i, C(τ), q, M, C×, phase classification, ω0
  theory/analytic.py             T_c, Σ(η), R_∞, Sompolinsky–Crisanti, RS static DMFT, stability
  theory/symbolic.py             13 SymPy checks (2 correct typos of the paper)
  theory/complexity.py           exact enumeration (fixed points, 2-cycles), NMFE multistart, Σ_2(α,η) saddle
  theory/dmft.py                 discrete-time DMFT, Monte-Carlo + Novikov response
  optim/problem.py               planner problem (SK ground state), dimod BQM export
  optim/solvers.py               exact, greedy, SA, PT, τ-EO, SDP+GW, SK-game learning, dimod wrappers
  optim/quantum.py               QuantumAnnealingSolver (D-Wave clique / embedding composite) — the plug-in point
  optim/benchmark.py             scaling study: success probability, residual, TTS_99, breakdown size N*
  outputs/                       one class per computation and per figure (objects of the category)
  registry.py                    Output base class, registry, caching pipeline, category export
  mathcat.py                     math objects / morphisms / uses relation
  docgen.py                      regenerates docs/02-05
julia/SKGameCategory/            Catlab.jl module: presentations, DepGraph & Uses ACSets, derivations, impact analysis
category/                        exported category.json, .dot, .mmd, SKGamePaperLiteral.jl
output/<kind>/<Class>/           every output, routed by class (result.npz / scalars.json / meta.json / figures)
tests/                           pytest suite (unit, theory vs simulation, solvers, category invariants, integration)
```

## Install and run

```bash
pip install -e ".[dwave,test]"          # numpy scipy matplotlib sympy numba dimod dwave-samplers pytest
python -m skgame list                    # all output classes with their inputs
python -m skgame run --preset quick      # everything, ~6 min on 2 cores
python -m skgame run Fig11C2vsEps --preset paper      # one figure (and only its upstream), paper sizes
python -m skgame deps Fig17OrderParameter             # upstream / downstream / derivation
python -m skgame category                # export the category for Catlab
python -m skgame.docgen                  # regenerate docs 02-05
pytest                                   # 96 tests
```

`quick` presets are scaled-down versions of the paper's (N ≤ 512, shorter times, fewer samples); `paper` presets use the paper's sizes where stated (N up to 1024, t up to 10^8 — hours to days). Caching: an output's hash covers its parameters, its class source code and its inputs' hashes, so editing a class recomputes exactly its downstream closure.

## Catlab.jl module

```bash
python -m skgame category
julia --project=julia/SKGameCategory -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=julia/SKGameCategory julia/run_category.jl
```
Builds two `Presentation(FreeSymmetricMonoidalCategory)` (output classes; math objects), a `DepGraph` ACSet and a `Uses` span ACSet (Out ← Use → Mor), and provides `derivation`, `upstream`, `downstream`, `topo_order`, `impacted_by`, `check_category`, `write_graphviz`. **Not executed in the build environment** (Julia could not be downloaded there); the same invariants are checked on the same JSON by `tests/test_registry_category.py`. Run `Pkg.test()` once locally.

## Planner benchmark and quantum annealing

`PlannerBenchmark` solves max_S (1/N) Sᵀ J S for N on a grid with every solver, against a reference that is exact (Gray code) up to `n_exact` and best-known (all solvers + a long parallel-tempering run) beyond. It reports per solver and N: median success probability, residual, TTS₉₉, the breakdown size N* (first N with median success < 0.5) and the SDP certification gap. To add the QPU:

```python
from skgame.optim.quantum import QuantumAnnealingSolver
from skgame.optim.solvers import default_solvers
from skgame.optim.benchmark import BenchmarkConfig, run_benchmark
solvers = default_solvers("paper") + [QuantumAnnealingSolver(num_reads=1000, annealing_time=20, clique=True)]
res = run_benchmark(solvers, BenchmarkConfig(Ns=(16, 32, 64, 96, 128, 160), n_instances=10, n_runs=5))
```
`clique=True` uses `DWaveCliqueSampler` (dense SK instances need a clique embedding; ~170 logical spins on Advantage). Any dimod sampler can be passed through `sampler_factory=`. Use `use_simulator=True` to test the plumbing without a token.

Caveat on "breakdown": above `n_exact` the reference is only best-known, so N* measures when a solver stops matching the best of the ensemble within its budget, not when it stops finding the true optimum. To push certification further, add an exact branch-and-bound / SDP-based solver (not included).

## Corrections to the paper found while reproducing

1. **Eq. (B36)**: the fixed-point complexity is written with $-x^2/2$; the form consistent with Eq. (23) (and giving Tanaka–Edwards' 0.1992) is Eq. (22), $-\eta x^2/2$.
2. **Eq. (49)**: the instability criterion must read $\beta(1-\varepsilon)\chi = 1-\beta\sqrt{\upsilon}$; only then does it reproduce $T_c(\varepsilon)$ of Eq. (20). The literal version gives a residual of 0.5 at $\varepsilon=0$.
3. **Fig. 15 convention**: the frequency $\omega_0$ is in cycles per step (the fundamental and its third harmonic both fall below $\omega/\sqrt\alpha=1$).

## Reproduction notes

* Fig. 4: the Sommers–Dupont curve for SK ground states is not recomputed (full-RSB local-field distribution); the slopes κ=1.6 (learning) and κ_SK≈0.6 are drawn as references.
* Fig. 10: the L=2 saddle is solved under the 2-cycle symmetric ansatz with continuation in α; it reaches α=0.02 and shows the coalescence Σ₂→Σ and 1−C(1)→0 that the paper could only extrapolate.
* Fig. 17: the static RS solution depends on the branch of m(z) when β(1−ε)χ>1; the outer (stable) branch is used and is stated in the code. With it the static curve lies close to the simulations; the instability (panel b) is branch-independent near T_c.
* Figs. 7, 14, 18 are the most sensitive to sizes and times; the `quick` preset shows the trends, the `paper` preset is needed for the exponents.
