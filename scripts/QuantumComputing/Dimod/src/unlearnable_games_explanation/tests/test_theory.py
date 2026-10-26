import numpy as np
import pytest

from skgame.model import InteractionEnsemble, SKGame, GameParams, WindowRecorder, eta, INF
from skgame.observables import stationary_autocorr
from skgame.theory import analytic as A
from skgame.theory import symbolic
from skgame.theory.complexity import (count_fixed_points_T0, count_two_cycles_T0, count_nmfe_solutions,
                                      Sigma2_saddle, Sigma2_curve, bvn_cdf_diag, annealed_complexity)
from skgame.theory.dmft import DMFTSolver


def test_all_symbolic_checks_pass():
    rows = symbolic.run_all()
    bad = [r for r in rows if not r["passed"]]
    assert not bad, bad


def test_Tc_values_and_two_routes():
    assert A.Tc(0.0) == pytest.approx(2.0)
    assert A.Tc(1.0) == pytest.approx(1 / np.sqrt(2))
    assert A.Tc(2.0) == pytest.approx(0.0)
    for e in [0.0, 0.3, 0.7, 1.0, 1.4, 1.8]:
        assert A.Tc_from_stability(e) == pytest.approx(A.Tc(e), rel=1e-8)


def test_lambda_max_matches_random_matrix():
    rng = np.random.default_rng(0)
    for e in [0.0, 0.5]:
        J = InteractionEnsemble(1500, e).sample(rng, 1)[0]
        lam = np.max(np.linalg.eigvals(J).real)
        assert lam == pytest.approx(A.lambda_max_elliptic(e), abs=0.06)


def test_complexity_T0():
    S, x = A.complexity_fp_T0(1.0)
    assert S == pytest.approx(0.19923, abs=1e-4)
    assert A.complexity_fp_T0(0.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_exact_counts_match_bruteforce():
    rng = np.random.default_rng(1)
    J = InteractionEnsemble(10, 0.3).sample(rng, 1)[0]
    n_fp = n_2c = 0
    states = [np.array([1 if (k >> i) & 1 else -1 for i in range(10)], float) for k in range(1024)]
    for S in states:
        h = J @ S
        if np.all(S * h > 0):
            n_fp += 1
        Sp = np.sign(h)
        if np.all(h != 0) and not np.array_equal(Sp, S) and np.all(S * (J @ Sp) > 0):
            n_2c += 1
    assert count_fixed_points_T0(J) == n_fp
    assert count_two_cycles_T0(J) == n_2c // 2


def test_exact_complexity_close_to_annealed_formula():
    rng = np.random.default_rng(2)
    N = 16
    counts = [count_fixed_points_T0(J) for J in InteractionEnsemble(N, 0.0).sample(rng, 40)]
    assert annealed_complexity(counts, N) == pytest.approx(0.199, abs=0.05)


def test_nmfe_count_paramagnet_above_Tc():
    J = InteractionEnsemble(10, 0.0).sample(np.random.default_rng(3), 1)[0]
    n, sols = count_nmfe_solutions(J, beta=0.3, n_starts=30, rng=0)
    assert n == 1 and np.allclose(sols[0], 0, atol=1e-6)


def test_bvn_diag():
    from scipy.stats import multivariate_normal as mvn
    for z, r in [(0.3, 0.4), (-1.0, -0.6), (0.0, 0.9), (2.0, 0.1)]:
        assert bvn_cdf_diag(z, r) == pytest.approx(mvn(cov=[[1, r], [r, 1]]).cdf([z, z]), abs=1e-6)


def test_sigma2_alpha1_and_alpha0():
    for e in [0.0, 0.4]:
        S, x, ok = Sigma2_saddle(1.0, eta(e))
        assert ok and S == pytest.approx(2 * A.complexity_fp_T0(eta(e))[0], rel=1e-8)
    c = Sigma2_curve(np.linspace(1, 0.03, 60), 1.0)
    assert c[-1, 1] == pytest.approx(A.complexity_fp_T0(1.0)[0], abs=0.01)   # coalescence alpha->0
    assert c[-1, 2] < 0.02                                                   # 1-C(1) -> 0


def test_sompolinsky_crisanti():
    tau, C, D = A.sompolinsky_crisanti()
    assert C[0] == pytest.approx(1.0)
    assert A.TAU1 == pytest.approx(2.8395, abs=1e-3)
    k = (tau > 5) & (tau < 10)
    slope = np.polyfit(tau[k], np.log(C[k]), 1)[0]
    assert slope == pytest.approx(-1 / A.TAU1, rel=0.03)


def test_static_rs_paramagnet_and_low_T():
    assert A.static_dmft_fixed_point(0.4, 0.0)[0] == pytest.approx(0.0, abs=1e-6)
    q, chi = A.static_dmft_fixed_point(5.0, 0.0, q0=0.9)
    assert 0.8 < q < 1.0


def test_dmft_matches_simulation_alpha1():
    for eps in [0.85, 1.05]:
        r = DMFTSolver(1.0, np.inf, eps, T=40, M=8000, n_iter=15, seed=1).solve()
        Cd = r.stationary_C(4, 15)
        g = SKGame(GameParams(N=256, alpha=1.0, eps=eps), R=8, seed=2)
        w = WindowRecorder(800, 100)
        g.run(900, [w])
        Cs = stationary_autocorr(w.result()["S"], 4).mean(0)
        assert np.max(np.abs(Cd - Cs)) < 0.08


def test_dmft_eps1_has_no_memory_term_and_decays():
    r = DMFTSolver(0.2, np.inf, 1.0, T=80, M=6000, n_iter=8).solve()
    assert np.allclose(r.G, 0)
    C = r.stationary_C(30, 60)
    assert C[0] == pytest.approx(1.0) and C[-1] < 0.3
