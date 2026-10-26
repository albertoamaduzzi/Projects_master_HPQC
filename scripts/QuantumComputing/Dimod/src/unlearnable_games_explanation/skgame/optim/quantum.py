"""Plug-in point for quantum annealing (single solver, used by the planner benchmark).

For the Zephyr / Pegasus comparison and every embedding knob (strategy, chain strength,
chain-break method, polishing) use `skgame.optim.backends.AnnealBackend`.

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

import time
from typing import Any, Callable

import numpy as np

from .problem import PlannerProblem
from .solvers import DimodSamplerSolver


class QuantumAnnealingSolver(DimodSamplerSolver):
    name = "quantum_annealing"

    def __init__(self, num_reads: int = 10000, annealing_time: float = 20.0, clique: bool = True,
                 use_simulator: bool = False, extra_kwargs: dict | None = None,
                 sampler_factory: Callable[[], Any] | None = None, solver_name: str | None = None,
                 max_N: int | None = 170, chain_strength: float | None = None):
        kw = {"num_reads": num_reads}
        if not use_simulator:
            kw["annealing_time"] = annealing_time
            if chain_strength is not None:
                kw["chain_strength"] = chain_strength
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

    @property
    def sampler(self):
        if self._sampler is None:
            self._sampler = self._factory()
        return self._sampler

    def largest_problem(self) -> int | None:
        """Largest N the sampler accepts (clique size on the QPU; None = no limit). No QPU time."""
        return getattr(self.sampler, "largest_clique_size", None)

    def solver_id(self) -> str:
        child = getattr(self.sampler, "child", self.sampler)
        solver = getattr(child, "solver", None)
        return getattr(solver, "name", type(self.sampler).__name__)

    def sample_problem(self, P: PlannerProblem, rng=None) -> dict:
        """One sampler call with every read kept: what a QPU benchmark needs.

        Returns the distinct rewards found and how many reads gave each (so the success
        probability per read can be computed later against any reference), the best
        configuration, wall time, and the QPU timing / chain-break statistics when present.
        Rewards are recomputed from the spins, not taken from the sampler's energies, so they
        are exact in the units of R_N whatever scaling the sampler applied internally.
        """
        rng = np.random.default_rng(rng)
        kw = dict(self.sample_kwargs)
        params = getattr(self.sampler, "parameters", None)
        if params is None or "seed" in params:
            kw.setdefault("seed", int(rng.integers(2**31)))
        t0 = time.perf_counter()
        ss = self.sampler.sample(P.to_bqm(), **kw)
        ss.resolve()
        wall = time.perf_counter() - t0
        from .backends import sampleset_statistics
        out = sampleset_statistics(ss, P, wall)
        out["info"].update(solver=self.solver_id(), sample_kwargs={k: v for k, v in kw.items() if k != "seed"})
        return out
