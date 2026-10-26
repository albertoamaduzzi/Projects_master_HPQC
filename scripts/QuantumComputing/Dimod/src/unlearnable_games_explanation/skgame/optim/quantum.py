"""Plug-in point for quantum annealing.

`QuantumAnnealingSolver` is a `DimodSamplerSolver` whose sampler factory builds a D-Wave QPU
composite (EmbeddingComposite(DWaveSampler()) by default). The planner problem is a dense
SK instance (complete graph K_N), so on Pegasus/Zephyr hardware it needs a clique embedding;
`DWaveCliqueSampler` is used when `clique=True` (recommended, N up to ~170 on Advantage).

Register your own variant by subclassing and overriding `make_sampler`; or pass any dimod
sampler factory. Nothing here is imported unless the solver is used, so the package and its
tests run without dwave-system or a Leap token. For tests, `use_simulator=True` swaps the
QPU for the classical simulated-annealing sampler with the same interface.

Recommended hardware knobs (map directly to sample kwargs): num_reads, annealing_time,
chain_strength (or a relative chain strength via dwave.embedding.chain_strength.uniform_torque_compensation),
auto_scale. Energies are in the units of H(S) = -sum_{i<j} W_ij S_i S_j, so
reward = -2 E / N (+ diag offset) and the benchmark compares on reward.
"""
from __future__ import annotations

from typing import Any, Callable

from .solvers import DimodSamplerSolver


class QuantumAnnealingSolver(DimodSamplerSolver):
    name = "quantum_annealing"

    def __init__(self, num_reads: int = 10000, annealing_time: float = 20.0, clique: bool = True,
                 use_simulator: bool = False, extra_kwargs: dict | None = None,
                 sampler_factory: Callable[[], Any] | None = None, solver_name: str | None = None,
                 max_N: int | None = 170):
        kw = {"num_reads": num_reads}
        if not use_simulator:
            kw["annealing_time"] = annealing_time
        kw.update(extra_kwargs or {})
        self.clique, self.use_simulator, self.solver_name = clique, use_simulator, solver_name
        super().__init__(sampler_factory or self.make_sampler,
                         "qa_simulated" if use_simulator else "quantum_annealing", kw, max_N=max_N)

    def make_sampler(self):
        if self.use_simulator:
            from dwave.samplers import SimulatedAnnealingSampler
            return SimulatedAnnealingSampler()
        if self.clique:
            from dwave.system import DWaveCliqueSampler
            return DWaveCliqueSampler(solver=self.solver_name) if self.solver_name else DWaveCliqueSampler()
        from dwave.system import DWaveSampler, EmbeddingComposite
        base = DWaveSampler(solver=self.solver_name) if self.solver_name else DWaveSampler()
        return EmbeddingComposite(base)
