import numpy as np
import pytest

from skgame.model import (InteractionEnsemble, GameParams, SKGame, RewardRecorder, WindowRecorder,
                          SnapshotRecorder, TwoTimeRecorder, upsilon, eta, eps_from_eta, INF)
from skgame.observables import (average_reward, individual_rewards, stationary_autocorr, cross_overlap,
                                detect_period, classify_phase, dominant_frequency)


@pytest.mark.parametrize("eps", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_ensemble_moments(eps):
    N, R = 200, 20
    J = InteractionEnsemble(N, eps).sample(np.random.default_rng(0), R)
    iu = np.triu_indices(N, 1)
    a, b = J[:, iu[0], iu[1]].ravel(), J[:, iu[1], iu[0]].ravel()
    assert np.allclose(J[:, np.arange(N), np.arange(N)], 0)
    assert abs(N * np.var(np.r_[a, b]) - upsilon(eps)) < 0.03
    if upsilon(eps) > 0:
        assert abs(np.corrcoef(a, b)[0, 1] - eta(eps)) < 0.02


def test_eta_inverse():
    for e in np.linspace(0, 2, 21):
        assert abs(eps_from_eta(eta(e)) - e) < 1e-9 or (e > 1 and abs(eta(eps_from_eta(eta(e))) - eta(e)) < 1e-9)


def test_symmetric_and_antisymmetric_limits():
    rng = np.random.default_rng(1)
    J0 = InteractionEnsemble(30, 0.0).sample(rng, 1)[0]
    J2 = InteractionEnsemble(30, 2.0).sample(rng, 1)[0]
    assert np.allclose(J0, J0.T) and np.allclose(J2, -J2.T)


def test_reward_identity_antisymmetric_drops():
    rng = np.random.default_rng(2)
    J = InteractionEnsemble(40, 0.7).sample(rng, 3)
    S = np.where(rng.random((3, 40)) < 0.5, 1.0, -1.0)
    Js = 0.5 * (J + np.swapaxes(J, 1, 2))
    assert np.allclose(average_reward(S, J), average_reward(S, Js))
    assert np.allclose(individual_rewards(S, J).mean(1), average_reward(S, J))


def test_noiseless_alpha1_is_hopfield_parallel():
    p = GameParams(N=50, alpha=1.0, eps=0.3)
    g = SKGame(p, R=2, seed=3)
    S0 = g.intentions(g.Q).copy()
    g.run(1)
    S1 = g.intentions(g.Q)
    assert np.array_equal(S1, np.where(np.einsum("rij,rj->ri", g.J, S0) >= 0, 1.0, -1.0))


def test_ewma_of_rewards():
    """Q(t) = (1-a)^t Q(0) + a sum (1-a)^{t-1-s} R(s): direct check of Eq. 3."""
    p = GameParams(N=20, alpha=0.3, beta=2.0, eps=0.5)
    g = SKGame(p, R=1, seed=4)
    Q0 = g.Q.copy()
    rec = []

    class Grab(RewardRecorder):
        def update(self, t, Q, m, S, R):
            rec.append(R.copy())
    g.run(10, [Grab()])
    a = p.alpha
    Q = (1 - a) ** 10 * Q0 + a * sum((1 - a) ** (9 - s) * rec[s] for s in range(10))
    assert np.allclose(Q, g.Q)


def test_logit_rule_statistics():
    p = GameParams(N=1, alpha=0.5, beta=1.3)
    g = SKGame(p, R=1, seed=5, J=np.zeros((1, 1)))
    m = np.full((200000, 1), np.tanh(0.7))
    S = g.decisions(m)
    assert abs(S.mean() - np.tanh(0.7)) < 0.01


def test_recorders_shapes():
    g = SKGame(GameParams(N=16, alpha=0.5, beta=3.0, eps=0.2), R=3, seed=6)
    w = WindowRecorder(5, 10, store_m=True, store_e=True)
    sn = SnapshotRecorder([0, 3, 7])
    tt = TwoTimeRecorder([2, 4], [1, 3])
    g.run(20, [w, sn, tt])
    r = w.result()
    assert r["m"].shape == (3, 10, 16) and r["S"].shape == (3, 10, 16) and r["e"].shape == (3, 10, 16)
    assert sn.result()["m"].shape == (3, 3, 16)
    assert np.all(np.isfinite(tt.result()["C"]))


def test_autocorr_and_periods():
    T, N = 40, 8
    x = np.ones((1, T, N))
    x[0, 1::2, :4] = -1             # half spins flip: C(odd)=0, C(even)=1  -> L = 2
    C = stationary_autocorr(x, 6)[0]
    assert np.allclose(C[::2], 1) and np.allclose(C[1::2], 0)
    assert detect_period(C) == 2
    assert classify_phase(np.ones((2, 20, 5)))[0] == 1


def test_cross_overlap_independent_vs_identical():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((50, 400))
    assert np.allclose(cross_overlap(a, a), a.var(1))
    assert abs(cross_overlap(a, rng.standard_normal((50, 400))).mean()) < 0.02


def test_dominant_frequency():
    t = np.arange(400)
    assert abs(dominant_frequency(np.cos(2 * np.pi * t / 25)) - 1 / 25) < 2e-3


def test_fixed_point_reached_symmetric_slow_learning():
    g = SKGame(GameParams(N=64, alpha=0.05, eps=0.0), R=4, seed=8)
    w = WindowRecorder(4000, 50)
    g.run(4050, [w])
    assert np.all(classify_phase(w.result()["S"]) == 1)
    S = g.intentions(g.Q)
    e = individual_rewards(S, g.J)
    assert np.all(e > 0)              # fixed point of a symmetric game: all rewards positive (Eq. 16)
