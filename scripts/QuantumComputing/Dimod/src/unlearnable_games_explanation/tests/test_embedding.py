"""Embeddings of K_N on Zephyr and Pegasus, the embedded simulator back-end and the cache."""
import numpy as np
import pytest

pytest.importorskip("minorminer")
pytest.importorskip("dwave.system")

from skgame.optim.backends import AnnealBackend, AnnealSettings, label, parse_spec  # noqa: E402
from skgame.optim.embedding import (CliqueEmbedding, EmbeddingCache, Topology, chain_stats,  # noqa: E402
                                    is_valid_embedding, topology)
from skgame.optim.problem import PlannerProblem  # noqa: E402
from skgame.optim.solvers import ExactEnumeration  # noqa: E402

SMALL = {"zephyr": Topology("zephyr", (2, 4), "Advantage2"), "pegasus": Topology("pegasus", (4,), "Advantage")}


@pytest.mark.parametrize("name", ["zephyr", "pegasus"])
def test_clique_embedding_is_valid_and_made_of_two_segments(name):
    G = SMALL[name].ideal_graph()
    emb = CliqueEmbedding().embed(16, G)
    assert is_valid_embedding(emb, 16, G)
    assert chain_stats(emb)["n_qubits"] == sum(len(c) for c in emb.values())
    broken = dict(emb)
    broken[1] = broken[0]                      # two spins on the same qubits: invalid
    assert not is_valid_embedding(broken, 16, G)


def test_production_shapes_zephyr_has_shorter_chains():
    z, p = topology("zephyr").ideal_graph(), topology("pegasus").ideal_graph()
    sz, sp = chain_stats(CliqueEmbedding().embed(128, z)), chain_stats(CliqueEmbedding().embed(128, p))
    assert sz["chain_len_max"] < sp["chain_len_max"]
    assert topology("zephyr").solver_filter == {"topology__type": "zephyr"}


def test_cache_roundtrip(tmp_path):
    G = SMALL["zephyr"].ideal_graph()
    cache = EmbeddingCache(tmp_path)
    a = cache.get("g", CliqueEmbedding(), 10, G)
    assert cache.path("g", "clique", 10).exists()
    assert cache.get("g", CliqueEmbedding(), 10, None) == a       # read back without the graph


def test_specs():
    assert parse_spec("qpu:zephyr:clique") == ("qpu", "zephyr", "clique")
    assert parse_spec("sim:logical") == ("sim", None, None)
    assert label("qpu:pegasus:clique") == "qpu_pegasus_clique"
    for bad in ("qpu:zephyr", "gpu:zephyr:clique", "qpu:chimera2:clique", "qpu:zephyr:magic"):
        with pytest.raises((ValueError, KeyError)):
            parse_spec(bad)


@pytest.mark.parametrize("method", ["majority_vote", "minimize_energy"])
def test_embedded_simulator_finds_small_ground_state(tmp_path, method):
    P = PlannerProblem.random(10, rng=3)
    be = AnnealBackend("sim:zephyr:clique", AnnealSettings(sim_num_reads=50, chain_break_method=method),
                       cache_root=tmp_path)
    r = be.sample(P, rng=0)
    opt = ExactEnumeration().solve(P).reward
    assert sum(r["counts"]) == 50
    assert r["polished_best_reward"] == pytest.approx(opt)
    assert r["info"]["chain_strength"] == pytest.approx(1.414 * np.sqrt(np.mean(
        [v ** 2 for v in P.to_bqm().quadratic.values()])) * np.sqrt(9))
    assert 0.0 <= r["info"]["chain_break_fraction_mean"] <= 1.0


def test_qubo_and_ising_are_the_same_problem():
    import itertools
    import dimod
    P = PlannerProblem.random(9, rng=2)
    Q, off = P.to_qubo()
    qubo = dimod.BinaryQuadraticModel.from_qubo(Q, offset=off)
    ising = P.to_bqm()
    for bits in itertools.product([0, 1], repeat=9):
        x = dict(enumerate(bits))
        S = {i: 2 * b - 1 for i, b in x.items()}
        assert qubo.energy(x) == pytest.approx(ising.energy(S))
        assert ising.energy(S) == pytest.approx(P.energy(np.array([S[i] for i in range(9)])))
    assert qubo.change_vartype(dimod.SPIN, inplace=False).is_almost_equal(ising)


@pytest.mark.parametrize("method", ["majority_vote", "discard"])
def test_per_read_broken_chains_are_recorded(tmp_path, method):
    P = PlannerProblem.random(20, rng=4)
    be = AnnealBackend("sim:zephyr:clique", AnnealSettings(sim_num_reads=120, sim_num_sweeps=50,
                       chain_strength_prefactor=0.3, chain_break_method=method), cache_root=tmp_path)
    r = be.sample(P, rng=0)
    hist = {int(k): v for k, v in r["info"]["broken_chains_hist"].items()}
    assert sum(hist.values()) == 120                                   # every physical read counted
    assert sum(r["info"]["chain_break_count"]) == sum(k * v for k, v in hist.items())
    t = r["reads"]
    assert sum(t["count"]) == 120 - r["info"]["discarded_reads"]
    assert len(t["reward"]) == len(t["n_broken"]) == len(t["polished_reward"])
    if method == "discard":
        assert set(t["n_broken"]) <= {0} and r["info"]["discarded_reads"] == 120 - hist.get(0, 0)
    else:
        assert max(t["n_broken"]) == max(hist) and r["info"]["discarded_reads"] == 0
        assert r["info"]["chain_break_fraction_mean"] == pytest.approx(
            sum(k * v for k, v in hist.items()) / (120 * 20))
