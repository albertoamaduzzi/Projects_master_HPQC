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
    def sample_kwargs(self) -> dict:
        """Keyword arguments for the (physical) sampler itself."""
        s = self.settings
        if self.device == "qpu":
            return {"num_reads": s.num_reads, "annealing_time": s.annealing_time}
        return {"num_reads": s.sim_num_reads, "num_sweeps": s.sim_num_sweeps}

    def sample(self, P: PlannerProblem, rng=None) -> dict:
        """One call. For embedded back-ends this does explicitly what FixedEmbeddingComposite does
        (embed -> sample -> unembed), so that the PHYSICAL reads are available in between: from
        them `broken_chains` tells, read by read, which logical spins had a broken chain."""
        rng = np.random.default_rng(rng)
        bqm = P.to_bqm()
        kw = self.sample_kwargs()
        params = getattr(self.child, "parameters", {}) or {}
        if self.device == "sim" and "seed" in params:
            kw["seed"] = int(rng.integers(2**31))
        broken, phys_counts, cs_used, emb = None, None, None, None
        t0 = time.perf_counter()
        if self.embedded:
            import dimod
            from dwave.embedding import EmbeddedStructure, broken_chains, unembed_sampleset
            from dwave.embedding.chain_strength import uniform_torque_compensation
            emb = self.embedding(P.N)
            structure = EmbeddedStructure(self.child.edgelist, emb)
            target = structure.embed_bqm(bqm, chain_strength=partial(
                uniform_torque_compensation, prefactor=self.settings.chain_strength_prefactor),
                smear_vartype=dimod.SPIN)
            cs_used = structure.chain_strength
            phys = self.child.sample(target, **kw)
            phys.resolve()
            chains = [emb[i] for i in range(P.N)]
            broken = np.asarray(broken_chains(phys, chains), bool)          # (physical rows, N)
            phys_counts = np.asarray(phys.record.num_occurrences, int)
            ss = unembed_sampleset(phys, structure, bqm,
                                   chain_break_method=_chain_break_method(self.settings.chain_break_method, bqm, emb),
                                   chain_break_fraction=True)
            ss.info.update({k: v for k, v in phys.info.items() if k not in ss.info})
        else:
            ss = self.child.sample(bqm, **kw)
            ss.resolve()
        wall = time.perf_counter() - t0
        out = sampleset_statistics(ss, P, wall, polish=self.settings.polish, broken=broken, phys_counts=phys_counts,
                                   discard=self.settings.chain_break_method == "discard")
        info = out["info"]
        info.update(backend=self.spec, solver=self.solver_id(), settings=asdict(self.settings),
                    sample_kwargs={k: v for k, v in kw.items() if k != "seed"})
        if emb is not None:
            info["embedding"] = chain_stats(emb)
            info["chain_lengths"] = [len(emb[i]) for i in range(P.N)]
            info["chain_strength"] = float(cs_used) if np.isscalar(cs_used) else None
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


def sampleset_statistics(ss, P: PlannerProblem, wall: float, polish: bool = False, broken=None,
                         phys_counts=None, discard: bool = False) -> dict:
    """Distinct rewards and their read counts (best first), best spins, QPU timing, chain breaks.

    Rewards are recomputed from the spins, so they are exact in units of R_N whatever scaling
    the sampler applied. With `polish`, every distinct read is also relaxed by greedy single-flip
    ascent (a classical post-processing step that repairs the residual errors of chain breaks
    and of analogue noise) and reported separately.

    With `broken` (physical rows x N booleans from dwave.embedding.broken_chains) the result also
    has, read by read, how many chains broke:
      out["reads"]                  distinct (configuration, number of broken chains) pairs:
                                    reward, polished_reward, n_broken, count
      info["broken_chains_hist"]    {number of broken chains: number of reads}
      info["chain_break_count"]     for each logical spin i, in how many reads its chain broke
      info["discarded_reads"]       reads dropped by chain_break_method="discard"
    """
    agg = ss.aggregate()
    if len(agg) == 0:                       # chain_break_method="discard" dropped every read
        S = np.zeros((0, P.N))
    else:
        cols = [list(agg.variables).index(i) for i in range(P.N)]
        S = np.asarray(agg.record.sample, float)[:, cols]
    counts = np.asarray(agg.record.num_occurrences, int)
    rewards = np.atleast_1d(P.reward(S)) if len(S) else np.zeros(0)
    order = np.argsort(-rewards)
    info = {"wall_seconds": wall, "num_reads": int(counts.sum())}
    timing = ss.info.get("timing")
    if timing:
        info["timing_us"] = {k: float(v) for k, v in timing.items() if np.isscalar(v)}
    if "chain_break_fraction" in ss.record.dtype.names and len(ss):
        info["chain_break_fraction_mean"] = float(np.average(ss.record.chain_break_fraction,
                                                             weights=ss.record.num_occurrences))
    empty = len(S) == 0
    out = {"rewards": rewards[order].tolist() if not empty else [], "counts": counts[order].tolist(),
           "best_reward": float(rewards[order[0]]) if not empty else -np.inf,
           "best_S": _spin_string(S[order[0]]) if not empty else "", "info": info}
    polish_of = None
    if polish and empty:
        out.update(polished_rewards=[], polished_counts=[], polished_best_reward=-np.inf, polished_best_S="")
        polish_of = {}
    elif polish:
        from .solvers import _greedy
        Sp = np.array([_greedy(P.W, s.copy())[0] for s in S])
        rp = np.atleast_1d(P.reward(Sp))
        op = np.argsort(-rp)
        out.update(polished_rewards=rp[op].tolist(), polished_counts=counts[op].tolist(),
                   polished_best_reward=float(rp[op[0]]), polished_best_S=_spin_string(Sp[op[0]]))
        polish_of = {S[k].tobytes(): float(rp[k]) for k in range(len(S))}
    if broken is not None:
        nb_phys = broken.sum(axis=1)
        info["chain_break_count"] = (broken * phys_counts[:, None]).sum(axis=0).astype(int).tolist()
        hist = {}
        for nb, c in zip(nb_phys, phys_counts):
            hist[int(nb)] = hist.get(int(nb), 0) + int(c)
        info["broken_chains_hist"] = {str(k): v for k, v in sorted(hist.items())}
        kept = nb_phys == 0 if discard else np.ones(len(nb_phys), bool)
        info["discarded_reads"] = int(phys_counts[~kept].sum())
        nb_rows = nb_phys[kept]
        Sr = (np.asarray(ss.record.sample, float)[:, [list(ss.variables).index(i) for i in range(P.N)]]
              if len(ss) else np.zeros((0, P.N)))
        assert len(Sr) == len(nb_rows), "unembedded rows do not match the physical reads"
        table = {}
        for s_row, nb, c in zip(Sr, nb_rows, ss.record.num_occurrences):
            key = (s_row.tobytes(), int(nb))
            table[key] = table.get(key, 0) + int(c)
        keys = list(table)
        Sk = np.array([np.frombuffer(k[0]) for k in keys]).reshape(len(keys), P.N)
        rk = np.atleast_1d(P.reward(Sk)) if keys else np.zeros(0)
        out["reads"] = {"reward": rk.tolist(), "n_broken": [k[1] for k in keys], "count": [table[k] for k in keys],
                        "polished_reward": [polish_of[k[0]] for k in keys] if polish_of is not None else None}
    return out


def _spin_string(S) -> str:
    return "".join("+" if x > 0 else "-" for x in S)
