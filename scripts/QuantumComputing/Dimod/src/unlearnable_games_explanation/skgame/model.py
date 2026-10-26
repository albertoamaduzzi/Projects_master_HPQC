"""Core model of the SK-game (Garnier-Brun, Benzaquen, Bouchaud, arXiv:2312.12252).

Objects defined here
--------------------
InteractionEnsemble : the law of J  (Eq. 8-11 + diagonal J_d of Sec. III G)
GameParams          : (N, alpha, beta, eps, J0, Jd, online)
SKGame              : the learning map (Eq. 1-4), batched over R independent realisations
Recorder subclasses : streaming observables attached to a run

Dynamics (one time step t -> t+1, for every agent i):
    m_i(t)   = tanh(beta Q_i(t))                         (Eq. 2)
    S_i(t)   = +1 w.p. (1+m_i)/2, -1 otherwise           (Eq. 1)  [beta=inf: S = sign Q]
    R_i(t)   = sum_j J_ij S_j(t)                         (Eq. 4, 11)
    Q_i(t+1) = (1-alpha) Q_i(t) + alpha R_i(t)           (Eq. 3)
Offline learning (Eq. 7) replaces S_j(t) by m_j(t) in R_i(t).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

INF = float("inf")


# --------------------------------------------------------------------------------------
# Interaction ensemble
# --------------------------------------------------------------------------------------
def upsilon(eps: float) -> float:
    """N Var[J_ij] = (1-eps/2)^2 + (eps/2)^2 = 1 - eps + eps^2/2   (Eq. 9)."""
    return 1.0 - eps + 0.5 * eps**2


def eta(eps: float) -> float:
    """Correlation of J_ij and J_ji: (1-eps)/upsilon(eps)   (Eq. 10)."""
    return (1.0 - eps) / upsilon(eps)


def eps_from_eta(et: float) -> float:
    """Inverse of eta(eps) on the branch eps in [0, 2]. Solves eta*(1-e+e^2/2) = 1-e."""
    if abs(et) < 1e-15:
        return 1.0
    # eta e^2/2 + (1-eta) e + (eta-1) = 0
    a, b, c = et / 2.0, 1.0 - et, et - 1.0
    disc = np.sqrt(b * b - 4 * a * c)
    roots = [(-b + disc) / (2 * a), (-b - disc) / (2 * a)]
    roots = [r for r in roots if -1e-12 <= r <= 2 + 1e-12]
    return float(min(roots, key=lambda r: abs(r - 1)))


@dataclass(frozen=True)
class InteractionEnsemble:
    """J = (1-eps/2) J^S + (eps/2) J^A + J0/N + Jd * Id  (Eq. 8, 11, Sec. III G).

    J^S symmetric, J^A antisymmetric, independent Gaussian entries of variance sigma^2/N,
    zero diagonal (unless Jd != 0).
    """

    N: int
    eps: float = 0.0
    sigma: float = 1.0
    J0: float = 0.0
    Jd: float = 0.0

    def sample(self, rng: np.random.Generator, R: int = 1) -> np.ndarray:
        N = self.N
        s = self.sigma / np.sqrt(N)
        G1 = rng.standard_normal((R, N, N))
        G2 = rng.standard_normal((R, N, N))
        JS = (G1 + np.swapaxes(G1, 1, 2)) / np.sqrt(2.0) * s
        JA = (G2 - np.swapaxes(G2, 1, 2)) / np.sqrt(2.0) * s
        J = (1.0 - self.eps / 2.0) * JS + (self.eps / 2.0) * JA
        idx = np.arange(N)
        J[:, idx, idx] = 0.0
        if self.J0 != 0.0:
            J += self.J0 / N
        if self.Jd != 0.0:
            J[:, idx, idx] += self.Jd
        return J

    @staticmethod
    def symmetric_part(J: np.ndarray) -> np.ndarray:
        return 0.5 * (J + np.swapaxes(J, -1, -2))


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GameParams:
    N: int = 256
    alpha: float = 0.01
    beta: float = INF
    eps: float = 0.0
    J0: float = 0.0
    Jd: float = 0.0
    online: bool = True
    q0_scale: float = 1.0  # std of the random initial incentives Q_i(0)

    @property
    def noiseless(self) -> bool:
        return not np.isfinite(self.beta)

    def ensemble(self) -> InteractionEnsemble:
        return InteractionEnsemble(self.N, self.eps, 1.0, self.J0, self.Jd)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["beta"] = "inf" if self.noiseless else self.beta
        return d


# --------------------------------------------------------------------------------------
# Recorders (streaming observables)
# --------------------------------------------------------------------------------------
class Recorder:
    """Called after S(t) is drawn and R(t)=J S(t) computed, before the Q update."""

    def start(self, game: "SKGame", t_max: int) -> None:  # noqa: D401
        pass

    def update(self, t: int, Q: np.ndarray, m: np.ndarray, S: np.ndarray, R: np.ndarray) -> None:
        pass

    def result(self):
        return None


class RewardRecorder(Recorder):
    """Average reward R_N(t) = (1/N) sum_i S_i R_i  (Eq. 12), per realisation."""

    def __init__(self, stride: int = 1):
        self.stride = stride
        self.t, self.r = [], []

    def update(self, t, Q, m, S, R):
        if t % self.stride == 0:
            self.t.append(t)
            self.r.append(np.mean(S * R, axis=1))

    def result(self):
        return {"t": np.asarray(self.t), "reward": np.asarray(self.r)}  # (T, R)


class WindowRecorder(Recorder):
    """Stores m(t) and S(t) for t in [t0, t0+W) (float32 / int8)."""

    def __init__(self, t0: int, W: int, stride: int = 1, store_m: bool = True, store_S: bool = True,
                 store_e: bool = False):
        self.t0, self.W, self.stride = t0, W, stride
        self.store_m, self.store_S, self.store_e = store_m, store_S, store_e

    def start(self, game, t_max):
        Rn, N = game.R, game.p.N
        n = (self.W + self.stride - 1) // self.stride
        self.m = np.zeros((Rn, n, N), np.float32) if self.store_m else None
        self.S = np.zeros((Rn, n, N), np.int8) if self.store_S else None
        self.e = np.zeros((Rn, n, N), np.float32) if self.store_e else None
        self.k = 0

    def update(self, t, Q, m, S, R):
        if self.t0 <= t < self.t0 + self.W and (t - self.t0) % self.stride == 0:
            if self.m is not None:
                self.m[:, self.k] = m
            if self.S is not None:
                self.S[:, self.k] = S
            if self.e is not None:
                self.e[:, self.k] = S * R
            self.k += 1

    def result(self):
        return {"m": self.m, "S": self.S, "e": self.e, "t0": self.t0, "stride": self.stride}


class SnapshotRecorder(Recorder):
    """Stores m(t) (and individual rewards) at an arbitrary list of times."""

    def __init__(self, times: Iterable[int], store_e: bool = True):
        self.times = sorted(set(int(x) for x in times))
        self._set = set(self.times)
        self.store_e = store_e
        self.data_m, self.data_S, self.data_e = {}, {}, {}

    def update(self, t, Q, m, S, R):
        if t in self._set:
            self.data_m[t] = m.astype(np.float32).copy()
            self.data_S[t] = S.astype(np.int8).copy()
            if self.store_e:
                self.data_e[t] = (S * R).astype(np.float32)

    def result(self):
        ts = [t for t in self.times if t in self.data_m]
        out = {"t": np.asarray(ts), "m": np.stack([self.data_m[t] for t in ts], 1),
               "S": np.stack([self.data_S[t] for t in ts], 1)}
        if self.store_e:
            out["e"] = np.stack([self.data_e[t] for t in ts], 1)
        return out


class TwoTimeRecorder(Recorder):
    """Aging correlation C(t_w, t_w + t) = (1/N) sum_i m_i(t_w) m_i(t_w+t) for several t_w."""

    def __init__(self, t_ws: Sequence[int], lags: Sequence[int]):
        self.t_ws = list(t_ws)
        self.lags = np.asarray(sorted(set(int(x) for x in lags)))
        self.ref = {}
        self.C = None
        self.targets = {}
        for a, tw in enumerate(self.t_ws):
            for b, lag in enumerate(self.lags):
                self.targets.setdefault(tw + lag, []).append((a, b))

    def start(self, game, t_max):
        self.C = np.full((len(self.t_ws), len(self.lags), game.R), np.nan)

    def update(self, t, Q, m, S, R):
        if t in self.t_ws:
            self.ref[t] = m.copy()
        if t in self.targets:
            for a, b in self.targets[t]:
                tw = self.t_ws[a]
                if tw in self.ref:
                    self.C[a, b] = np.mean(self.ref[tw] * m, axis=1)

    def result(self):
        return {"t_w": np.asarray(self.t_ws), "lags": self.lags, "C": self.C}


# --------------------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------------------
@dataclass
class SKGame:
    """Batched simulator: R independent (J, initial condition) realisations in parallel."""

    p: GameParams
    R: int = 8
    seed: int = 0
    J: np.ndarray | None = None
    Q0: np.ndarray | None = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if self.J is None:
            self.J = self.p.ensemble().sample(self.rng, self.R)
        else:
            self.J = np.asarray(self.J, dtype=float)
            if self.J.ndim == 2:
                self.J = np.broadcast_to(self.J, (self.R,) + self.J.shape).copy()
        if self.Q0 is None:
            self.Q0 = self.p.q0_scale * self.rng.standard_normal((self.R, self.p.N))
        self.Q = np.array(self.Q0, dtype=float)
        self.t = 0

    # ---- elementary maps -------------------------------------------------------------
    def intentions(self, Q: np.ndarray) -> np.ndarray:
        if self.p.noiseless:
            return np.where(Q >= 0.0, 1.0, -1.0)
        return np.tanh(self.p.beta * Q)

    def decisions(self, m: np.ndarray) -> np.ndarray:
        if self.p.noiseless:
            return m
        u = self.rng.random(m.shape)
        return np.where(u < 0.5 * (1.0 + m), 1.0, -1.0)

    def rewards(self, X: np.ndarray) -> np.ndarray:
        # R_i = sum_j J_ij X_j, batched: (R,N,N) x (R,N) -> (R,N)
        return np.matmul(self.J, X[..., None])[..., 0]

    # ---- run ---------------------------------------------------------------------------
    def run(self, t_max: int, recorders: Sequence[Recorder] = ()) -> "SKGame":
        for rec in recorders:
            rec.start(self, t_max)
        a = self.p.alpha
        for _ in range(t_max):
            m = self.intentions(self.Q)
            S = self.decisions(m)
            X = S if self.p.online else m
            R = self.rewards(X)
            for rec in recorders:
                rec.update(self.t, self.Q, m, S, R)
            self.Q *= (1.0 - a)
            self.Q += a * R
            self.t += 1
        return self

    # ---- convenience --------------------------------------------------------------------
    def state(self):
        m = self.intentions(self.Q)
        return m


def ground_truth_reward_inf(eps: float) -> float:
    """R_inf = 0.7631... (2 - eps): Parisi value 2 * 0.76317 * (1 - eps/2)   (Eq. 13)."""
    return 0.763166726 * (2.0 - eps)


def finite_size_reward(N: int | np.ndarray, eps: float) -> np.ndarray:
    """R_N ~ R_inf - A/N^{2/3}, A ~ 0.75 (2-eps)   (footnote [59], Boettcher 2010)."""
    return ground_truth_reward_inf(eps) - 0.75 * (2.0 - eps) / np.asarray(N, float) ** (2.0 / 3.0)
