"""Minor embeddings of the planner problem (complete graph K_N) on D-Wave topologies.

The planner's BQM couples every pair of spins, so it is K_N. A QPU only couples the qubits that
are neighbours on its graph (Pegasus: degree 15, Zephyr: degree 20), so each logical spin i
becomes a *chain* of qubits C_i, connected on the hardware graph, held together by a strong
ferromagnetic coupling, with at least one hardware edge between every pair of chains.

Three independent pieces, combined by `skgame.optim.backends`:

  Topology            which hardware graph (ideal, generated offline; or the real solver's)
  EmbeddingStrategy   how K_N is mapped on it: `CliqueEmbedding` (native all-to-all, busclique)
                      or `HeuristicEmbedding` (minorminer, general purpose)
  EmbeddingCache      embeddings saved as JSON per (graph, strategy, N), so every call, every
                      job and every machine uses the same chains

See docs/06_qpu_chains_embedding.md for the theory and how to tune them.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import time

import numpy as np


# ======================================================================== topologies ===
@dataclass(frozen=True)
class Topology:
    """A D-Wave graph family with the shape of the current production QPU.

    `solver_filter` is what Leap needs to pick a QPU of this family; `ideal_graph()` builds the
    defect-free graph offline (used for planning and for the classical `sim` stand-in).
    """

    name: str
    shape: tuple
    qpu_family: str

    @property
    def solver_filter(self) -> dict:
        return {"topology__type": self.name}

    @property
    def graph_id(self) -> str:
        return f"ideal_{self.name}_" + "_".join(str(s) for s in self.shape)

    def ideal_graph(self):
        g = _graphs()
        if self.name == "zephyr":
            return g.zephyr_graph(*self.shape)
        if self.name == "pegasus":
            return g.pegasus_graph(*self.shape)
        if self.name == "chimera":
            return g.chimera_graph(*self.shape)
        raise ValueError(self.name)


def _graphs():
    """dwave-networkx was renamed dwave-graphs; accept either."""
    try:
        import dwave.graphs as g
    except ImportError:  # pragma: no cover
        import dwave_networkx as g
    return g


TOPOLOGIES = {
    # Advantage2 (Zephyr Z(12,4): 4800 nominal qubits, degree 20)
    "zephyr": Topology("zephyr", (12, 4), "Advantage2"),
    # Advantage (Pegasus P16: 5640 nominal qubits, degree 15)
    "pegasus": Topology("pegasus", (16,), "Advantage"),
}


def topology(name: str) -> Topology:
    try:
        return TOPOLOGIES[name]
    except KeyError:
        raise KeyError(f"unknown topology {name!r}; available: {sorted(TOPOLOGIES)}") from None


# ======================================================================== strategies ===
class EmbeddingStrategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def embed(self, N: int, target) -> dict:
        """Embedding {logical spin i: [qubits]} of K_N in the networkx graph `target` ({} if none)."""

    def max_size(self, target) -> int | None:
        """Largest N this strategy can embed in `target` (None: unknown, try and see)."""
        return None

    def describe(self) -> dict:
        return {"name": self.name}


class CliqueEmbedding(EmbeddingStrategy):
    """Native all-to-all embedding (minorminer.busclique): every chain has the same, minimal-ish
    length (Zephyr: about N/16 + 2 qubits; Pegasus: about N/12 + 2), deterministic and fast.
    This is what DWaveCliqueSampler uses internally."""

    name = "clique"
    _max_size_memo: dict = {}

    def embed(self, N, target):
        from minorminer.busclique import find_clique_embedding
        emb = _with_disk_cache(lambda: find_clique_embedding(N, target, use_cache=True),
                               lambda: _clique_no_cache(N, target))
        return {int(k): [int(q) for q in v] for k, v in emb.items()}

    def max_size(self, target):
        """Largest K_N of `target`, remembered in memory: the disk cache is read once per graph."""
        from minorminer.busclique import busgraph_cache
        key = _graph_key(target)
        if key not in self._max_size_memo:
            self._max_size_memo[key] = _with_disk_cache(lambda: len(busgraph_cache(target).largest_clique()),
                                                        lambda: _largest_clique_no_cache(target))
        return self._max_size_memo[key]


def _graph_key(g) -> tuple:
    return (g.number_of_nodes(), g.number_of_edges(), hash(frozenset(g.nodes())), hash(frozenset(map(frozenset, g.edges()))))


def _with_disk_cache(cached, uncached, retries: int = 4):
    """minorminer keeps its clique cache in files (inside the virtual environment). On Windows a synced
    folder (OneDrive), an antivirus or a second process can lock them: PermissionError on the rename of
    `.lru~`. Retry with back-off, then compute without the disk cache (slower, same result)."""
    for attempt in range(retries):
        try:
            return cached()
        except OSError as exc:
            wait = 0.5 * 2 ** attempt
            print(f"[embedding] clique cache locked ({type(exc).__name__}: {exc}); retry in {wait:g}s")
            time.sleep(wait)
    print("[embedding] clique cache still locked: computing without it")
    return uncached()


SMALLEST_UNCACHED = 12   # minorminer 0.2.22 aborts (std::bad_alloc) on uncached cliques of N <= 8


def _clique_no_cache(N: int, target) -> dict:
    """Clique embedding without the disk cache. For small N it embeds K_12 and keeps the first N chains:
    still a valid embedding of K_N (every chain connected, every pair of chains adjacent)."""
    from minorminer.busclique import find_clique_embedding
    M = max(N, SMALLEST_UNCACHED)
    emb = find_clique_embedding(M, target, use_cache=False)
    return {i: emb[i] for i in range(N)} if emb else {}


def _largest_clique_no_cache(target) -> int:
    """Largest N with a clique embedding, by bisection, without minorminer's disk cache."""
    from minorminer.busclique import find_clique_embedding
    # a clique of N needs about N^2 / 6 qubits or more on D-Wave graphs: N <= sqrt(8 n) is a safe bound
    # (asking minorminer for impossible sizes is slow and memory-hungry)
    if not find_clique_embedding(SMALLEST_UNCACHED, target, use_cache=False):
        return 0
    lo, hi = SMALLEST_UNCACHED, int(np.sqrt(8 * target.number_of_nodes())) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if find_clique_embedding(mid, target, use_cache=False):
            lo = mid
        else:
            hi = mid - 1
    return lo


class HeuristicEmbedding(EmbeddingStrategy):
    """General-purpose minorminer heuristic, best of `tries` seeds (shortest longest chain, then
    fewest qubits). Slower and usually worse than the clique embedding for K_N; kept as a
    baseline and for non-complete problems."""

    name = "minorminer"

    def __init__(self, tries: int = 4, timeout: float = 120.0, seed: int = 0):
        self.tries, self.timeout, self.seed = tries, timeout, seed

    def embed(self, N, target):
        import minorminer
        import networkx as nx
        src = list(nx.complete_graph(N).edges())
        best, key = {}, None
        for t in range(self.tries):
            emb = minorminer.find_embedding(src, target.edges(), random_seed=self.seed + t, timeout=self.timeout)
            if not emb:
                continue
            k = (max(len(c) for c in emb.values()), sum(len(c) for c in emb.values()))
            if key is None or k < key:
                best, key = emb, k
        return {int(k): [int(q) for q in v] for k, v in best.items()}

    def describe(self):
        return {"name": self.name, "tries": self.tries, "timeout": self.timeout, "seed": self.seed}


STRATEGIES = {"clique": CliqueEmbedding, "minorminer": HeuristicEmbedding}


def strategy(name: str, **kw) -> EmbeddingStrategy:
    try:
        return STRATEGIES[name](**kw)
    except KeyError:
        raise KeyError(f"unknown embedding strategy {name!r}; available: {sorted(STRATEGIES)}") from None


# ============================================================================= stats ===
def chain_stats(embedding: dict) -> dict:
    """Chain lengths of an embedding: the main predictor of chain breaks and of the energy
    compression imposed by the chain couplings."""
    if not embedding:
        return {"n_qubits": 0, "chain_len_mean": float("nan"), "chain_len_max": 0, "chain_len_min": 0}
    L = np.array([len(c) for c in embedding.values()])
    return {"n_qubits": int(L.sum()), "chain_len_mean": float(L.mean()),
            "chain_len_max": int(L.max()), "chain_len_min": int(L.min())}


def is_valid_embedding(embedding: dict, N: int, target) -> bool:
    """Every chain connected in `target`, chains disjoint, an edge between every pair of chains."""
    import networkx as nx
    if sorted(embedding) != list(range(N)):
        return False
    owner = {}
    for i, chain in embedding.items():
        if not nx.is_connected(target.subgraph(chain)):
            return False
        for q in chain:
            if q in owner:
                return False
            owner[q] = i
    linked = set()
    for u, v in target.edges():
        a, b = owner.get(u), owner.get(v)
        if a is not None and b is not None and a != b:
            linked.add((min(a, b), max(a, b)))
    return len(linked) == N * (N - 1) // 2


# ============================================================================= cache ===
class EmbeddingCache:
    """Embeddings on disk: <root>/<graph_id>/<strategy>/N0064.json.

    graph_id is the solver name for a real QPU (its working graph has missing qubits, so an
    embedding computed on the ideal graph may not fit) and `ideal_<topology>_<shape>` offline.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, graph_id: str, strategy_name: str, N: int) -> Path:
        return self.root / graph_id / strategy_name / f"N{N:04d}.json"

    def get(self, graph_id: str, strat: EmbeddingStrategy, N: int, target) -> dict:
        """Cached embedding, VALIDATED against `target` when a graph is given.

        A QPU's working graph changes over time (qubits and couplers go offline or come back),
        so an embedding cached on an earlier day can contain a chain that is no longer connected
        (DisconnectedChainError). Every cached embedding is therefore re-checked against the
        current graph; if it no longer fits it is recomputed and the file is replaced (the old one
        is kept as N....stale.json for the record)."""
        p = self.path(graph_id, strat.name, N)
        self.last_status = "computed"
        if p.exists():
            d = json.loads(p.read_text())
            emb = {int(k): v for k, v in d["embedding"].items()}
            if target is None or is_valid_embedding(emb, N, target):
                self.last_status = "cached"
                return emb
            os.replace(p, p.with_name(p.stem + ".stale.json"))
            self.last_status = "recomputed (cached one no longer fits the current graph)"
        emb = strat.embed(N, target)
        if emb:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({"graph_id": graph_id, "strategy": strat.describe(), "N": N,
                                       "stats": chain_stats(emb), "embedding": emb}))
            os.replace(tmp, p)
        return emb
