"""Annealing back-ends: device x topology x embedding strategy, behind one `sample(P)` call.

A back-end is named by a spec string

    qpu:zephyr:clique      D-Wave Advantage2 (Zephyr), all-to-all clique embedding
    qpu:pegasus:clique     D-Wave Advantage  (Pegasus), all-to-all clique embedding
    qpu:zephyr:minorminer  heuristic embedding instead of the clique one
    sim:zephyr:clique      the SAME embedded problem (chains included) solved by classical
                           simulated annealing on the ideal graph: no token, tests the pipeline
                           and shows the classical effect of the chains
    sim:logical            simulated annealing on K_N directly (no embedding)

so the Zephyr / Pegasus comparison is two back-ends run on the same instances with the same
`AnnealSettings`; nothing else changes. Each call returns every read (distinct rewards and
counts), the best configuration, the same after a greedy single-flip polish, the chain
statistics of the embedding, the chain strength used, the chain-break fraction and the QPU
timing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import numpy as np

from .embedding import EmbeddingCache, chain_stats, strategy as make_strategy, topology as make_topology
from .problem import PlannerProblem

CHAIN_BREAK_METHODS = ("majority_vote", "minimize_energy", "weighted_random", "discard")


@dataclass
class AnnealSettings:
    num_reads: int = 1000
    annealing_time: float = 20.0              # microseconds (QPU)
    chain_strength_prefactor: float = 1.414   # x uniform_torque_compensation(bqm, embedding)
    chain_break_method: str = "majority_vote"
    polish: bool = True                       # also report reads after greedy single-flip descent
    sim_num_reads: int = 100                  # reads when device == "sim"
    sim_num_sweeps: int = 1000
    solver_name: str | None = None            # pin a QPU by name; default: any QPU of the topology


def parse_spec(spec: str) -> tuple[str, str | None, str | None]:
    parts = spec.split(":")
    if parts == ["sim", "logical"]:
        return "sim", None, None
    if len(parts) != 3 or parts[0] not in ("qpu", "sim"):
        raise ValueError(f"bad back-end {spec!r}: use device:topology:strategy, e.g. qpu:zephyr:clique, "
                         "or sim:logical")
    make_topology(parts[1])
    make_strategy(parts[2])
    return parts[0], parts[1], parts[2]


def label(spec: str) -> str:
    """Directory-friendly name: qpu:zephyr:clique -> qpu_zephyr_clique."""
    return spec.replace(":", "_")


class AnnealBackend:
    def __init__(self, spec: str, settings: AnnealSettings | None = None, cache_root: Path | None = None,
                 child=None):
        self.spec = spec
        self.device, topo, strat = parse_spec(spec)
        self.settings = settings or AnnealSettings()
        self.topology = make_topology(topo) if topo else None
        self.strategy = make_strategy(strat) if strat else None
        self.cache = EmbeddingCache(cache_root) if cache_root is not None else None
        self._child = child                   # injectable (tests: MockDWaveSampler)
        self._target = None
        self._embeddings = {}

    @property
    def label(self) -> str:
        return label(self.spec)

    @property
    def embedded(self) -> bool:
        return self.topology is not None

    # ---------------------------------------------------------------- hardware graph ----
    @property
    def child(self):
        """The structured sampler the embedding maps onto."""
        if self._child is None:
            if self.device == "qpu":
                from dwave.system import DWaveSampler
                s = self.settings
                self._child = DWaveSampler(solver=s.solver_name or self.topology.solver_filter)
            elif self.embedded:
                import dimod
                from dwave.samplers import SimulatedAnnealingSampler
                G = self._target = self.topology.ideal_graph()
                self._child = dimod.StructureComposite(SimulatedAnnealingSampler(), list(G.nodes), list(G.edges))
            else:
                from dwave.samplers import SimulatedAnnealingSampler
                self._child = SimulatedAnnealingSampler()
        return self._child

    @property
    def target(self):
        """networkx graph of the working qubits and couplers."""
        if self._target is None and self.embedded:
            child = self.child                 # building a sim child also sets the ideal graph
            if self._target is not None:
                pass
            elif hasattr(child, "to_networkx_graph"):
                self._target = child.to_networkx_graph()   # real working graph, keeps the topology metadata
            else:
                raise ValueError(f"{self.spec}: cannot read the hardware graph of {type(child).__name__}")
        return self._target

    def graph_id(self) -> str:
        if self.device == "qpu":
            return self.solver_id()
        return self.topology.graph_id if self.embedded else "logical"

    def solver_id(self) -> str:
        solver = getattr(self.child, "solver", None)
        name = getattr(solver, "name", None)
        if isinstance(name, str):
            return name
        if self.embedded:
            return f"{type(self.child).__name__}_{self.topology.graph_id}"
        return type(self.child).__name__

    def largest_problem(self) -> int | None:
        return self.strategy.max_size(self.target) if self.embedded else None

    def embedding(self, N: int) -> dict:
        if N not in self._embeddings:
            if self.cache is not None:
                emb = self.cache.get(self.graph_id(), self.strategy, N, self.target)
            else:
                emb = self.strategy.embed(N, self.target)
            if not emb:
                raise ValueError(f"{self.spec}: no embedding of K_{N} on {self.graph_id()}")
            self._embeddings[N] = emb
        return self._embeddings[N]

    # -------------------------------------------------------------------- sampling ------
    def sample_kwargs(self, bqm, emb) -> dict:
        s = self.settings
        if self.device == "qpu":
            kw = {"num_reads": s.num_reads, "annealing_time": s.annealing_time}
        else:
            kw = {"num_reads": s.sim_num_reads, "num_sweeps": s.sim_num_sweeps}
        if self.embedded:
            from dwave.embedding.chain_strength import uniform_torque_compensation
            kw["chain_strength"] = partial(uniform_torque_compensation, prefactor=s.chain_strength_prefactor)
            kw["chain_break_method"] = _chain_break_method(s.chain_break_method, bqm, emb)
            kw["return_embedding"] = True
        return kw

    def sample(self, P: PlannerProblem, rng=None) -> dict:
        from dwave.system import FixedEmbeddingComposite
        rng = np.random.default_rng(rng)
        bqm = P.to_bqm()
        if self.embedded:
            emb = self.embedding(P.N)
            sampler = FixedEmbeddingComposite(self.child, emb)
        else:
            emb, sampler = None, self.child
        kw = self.sample_kwargs(bqm, emb)
        params = getattr(self.child, "parameters", {}) or {}
        if self.device == "sim" and "seed" in params:
            kw["seed"] = int(rng.integers(2**31))
        t0 = time.perf_counter()
        ss = sampler.sample(bqm, **kw)
        ss.resolve()
        wall = time.perf_counter() - t0
        out = sampleset_statistics(ss, P, wall, polish=self.settings.polish)
        info = out["info"]
        info.update(backend=self.spec, solver=self.solver_id(), settings=asdict(self.settings),
                    sample_kwargs={k: v for k, v in kw.items() if not callable(v) and k != "seed"})
        if emb is not None:
            info["embedding"] = chain_stats(emb)
            ctx = ss.info.get("embedding_context", {})
            cs = ctx.get("chain_strength")
            info["chain_strength"] = float(cs) if np.isscalar(cs) else None
            info["max_abs_logical_coupling"] = float(np.max(np.abs(P.W)))
        return out

    def estimate_access_time(self, N: int) -> float | None:
        """Seconds of QPU access time for one call, from the solver's timing model (no QPU use)."""
        solver = getattr(self.child, "solver", None)
        est = getattr(solver, "estimate_qpu_access_time", None)
        if est is None:
            return None
        n_qubits = chain_stats(self.embedding(N))["n_qubits"]
        return est(n_qubits, num_reads=self.settings.num_reads, annealing_time=self.settings.annealing_time) / 1e6


def _chain_break_method(name: str, bqm, emb):
    from dwave.embedding import chain_breaks as cb
    if name == "minimize_energy":
        resolver = cb.MinimizeEnergy(bqm, emb)

        def minimize_energy(samples, chains):    # unembed_sampleset needs a __name__
            return resolver(samples, chains)
        return minimize_energy
    if name in ("majority_vote", "weighted_random", "discard"):
        return getattr(cb, name)
    raise ValueError(f"chain_break_method must be one of {CHAIN_BREAK_METHODS}")


def sampleset_statistics(ss, P: PlannerProblem, wall: float, polish: bool = False) -> dict:
    """Distinct rewards and their read counts (best first), best spins, QPU timing, chain breaks.

    Rewards are recomputed from the spins, so they are exact in units of R_N whatever scaling
    the sampler applied. With `polish`, every distinct read is also relaxed by greedy single-flip
    ascent (a classical post-processing step that repairs the residual errors of chain breaks
    and of analogue noise) and reported separately.
    """
    agg = ss.aggregate()
    cols = [list(agg.variables).index(i) for i in range(P.N)]
    S = np.asarray(agg.record.sample, float)[:, cols]
    counts = np.asarray(agg.record.num_occurrences, int)
    rewards = np.atleast_1d(P.reward(S))
    order = np.argsort(-rewards)
    info = {"wall_seconds": wall, "num_reads": int(counts.sum())}
    timing = ss.info.get("timing")
    if timing:
        info["timing_us"] = {k: float(v) for k, v in timing.items() if np.isscalar(v)}
    if "chain_break_fraction" in ss.record.dtype.names:
        info["chain_break_fraction_mean"] = float(np.average(ss.record.chain_break_fraction,
                                                             weights=ss.record.num_occurrences))
    out = {"rewards": rewards[order].tolist(), "counts": counts[order].tolist(),
           "best_reward": float(rewards[order[0]]), "best_S": _spin_string(S[order[0]]), "info": info}
    if polish:
        from .solvers import _greedy
        Sp = np.array([_greedy(P.W, s.copy())[0] for s in S])
        rp = np.atleast_1d(P.reward(Sp))
        op = np.argsort(-rp)
        out.update(polished_rewards=rp[op].tolist(), polished_counts=counts[op].tolist(),
                   polished_best_reward=float(rp[op[0]]), polished_best_S=_spin_string(Sp[op[0]]))
    return out


def _spin_string(S) -> str:
    return "".join("+" if x > 0 else "-" for x in S)
