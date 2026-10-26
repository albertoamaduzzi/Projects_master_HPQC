# Visual explanations

Read in this order; each folder has a README with the step-by-step explanation and a script that regenerates every figure from the code in `../unlearnable_games_explanation/skgame`.

| folder | what it explains |
|---|---|
| [`SK_game/`](SK_game/README.md) | the SK game and its optimisation problem: couplings, local fields, satisfaction, energy, local minima, basins, phases of the learning dynamics, satisficing. Defines the **glossary** used everywhere else |
| [`optimal_algorithms/`](optimal_algorithms/README.md) | each optimiser of the benchmark (exact, greedy, game learning, SA, PT, τ-EO, tabu, SDP+GW, dimod samplers, quantum annealing), described with the glossary terms and shown acting on an SK instance |
| [`embedding_explanation/`](embedding_explanation/README.md) | how the all-to-all SK problem is embedded on D-Wave Zephyr and Pegasus (chains, step by step), and [how to tune chains and the QPU](embedding_explanation/chains_and_tuning.md) |

`common.py` holds the shared helpers (instances, energy, local fields, satisfaction, exhaustive landscape). Use the same Python environment as the campaign:

```bash
python SK_game/sk_game_explanation.py                     # ~1 min
python optimal_algorithms/algorithms_explanation.py       # ~1 min
python embedding_explanation/embedding_explanation.py     # ~30 s (--heuristic: +3 min)
```
