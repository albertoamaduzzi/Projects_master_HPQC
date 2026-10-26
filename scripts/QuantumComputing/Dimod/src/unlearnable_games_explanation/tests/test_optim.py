import numpy as np
import pytest

from skgame.optim.problem import PlannerProblem
from skgame.optim.solvers import (ExactEnumeration, GreedyMultistart, SimulatedAnnealing, ParallelTempering,
                                  ExtremalOptimization, SKGameLearner, SDPBurerMonteiro, default_solvers,
                                  DimodSamplerSolver)
from skgame.optim.quantum import QuantumAnnealingSolver
from skgame.optim.benchmark import BenchmarkConfig, run_benchmark


def brute(P):
    N = P.N
    best = -np.inf
    for k in range(2 ** N):
        S = np.array([1 if (k >> i) & 1 else -1 for i in range(N)], float)
        best = max(best, P.reward(S))
    return best


@pytest.fixture(scope="module")
def P12():
    return PlannerProblem.random(12, eps=0.5, rng=0)


def test_problem_reward_energy_consistency(P12):
    rng = np.random.default_rng(1)
    S = np.where(rng.random(12) < 0.5, 1.0, -1.0)
    assert P12.reward(S) == pytest.approx(P12.reward_from_energy(P12.energy(S)))
    bqm = P12.to_bqm()
    assert bqm.energy({i: S[i] for i in range(12)}) == pytest.approx(P12.energy(S))
    # reward uses only the symmetric part of J
    J = P12.J
    assert P12.reward(S) == pytest.approx(S @ J @ S / 12)


def test_exact_is_exact(P12):
    assert ExactEnumeration().solve(P12).reward == pytest.approx(brute(P12))


@pytest.mark.parametrize("solver", [GreedyMultistart(100), SimulatedAnnealing(sweeps=500), ParallelTempering(sweeps=300),
                                    ExtremalOptimization(steps_per_spin=200), SDPBurerMonteiro(), SKGameLearner(steps=500, restarts=8)])
def test_heuristics_reach_optimum_small(P12, solver):
    opt = brute(P12)
    r = solver.solve(P12, 3)
    assert r.reward <= opt + 1e-9
    if solver.name != "skgame_learning":
        assert r.reward == pytest.approx(opt)


def test_sdp_upper_bound(P12):
    r = SDPBurerMonteiro(iters=300).solve(P12, 0)
    assert r.info["upper_bound_reward"] >= brute(P12) - 1e-6


def test_dimod_wrappers_and_quantum_stub(P12):
    opt = brute(P12)
    for s in default_solvers("quick"):
        if isinstance(s, DimodSamplerSolver):
            assert s.solve(P12, 0).reward == pytest.approx(opt)
    qa = QuantumAnnealingSolver(num_reads=20, use_simulator=True)
    assert qa.solve(P12, 0).reward == pytest.approx(opt)


def test_custom_sampler_factory(P12):
    from dwave.samplers import SteepestDescentSolver
    s = QuantumAnnealingSolver(sampler_factory=SteepestDescentSolver, use_simulator=True, num_reads=200)
    assert s.solve(P12, 0).reward <= brute(P12) + 1e-9


def test_benchmark_small():
    cfg = BenchmarkConfig(Ns=(8, 12, 30), n_instances=2, n_runs=2, n_exact=12, ref_sweeps=200, ref_restarts=1)
    res = run_benchmark([ExactEnumeration(12), GreedyMultistart(5), SimulatedAnnealing(sweeps=100, restarts=1)], cfg,
                        log=lambda *a: None)
    m = res.metrics()
    assert res.ref_exact.tolist() == [True, True, False]
    assert np.all(m["p_median"][1:, :2] >= 0)
    assert m["p_median"][2, 0] == 1.0
    assert np.all(res.reward[1:][np.isfinite(res.reward[1:])] <= res.ref.max() + 1e-9)
