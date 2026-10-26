"""Solvers for the planner problem. Common interface:

    solver.solve(problem, rng) -> SolveResult(S, reward, seconds, evals, info)

`evals` counts single-spin-flip proposals (or equivalent work units), so solvers with very
different internal mechanics can be compared on work as well as wall time.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numba import njit

from .problem import PlannerProblem


@dataclass
class SolveResult:
    S: np.ndarray
    reward: float
    seconds: float
    evals: float
    info: dict = field(default_factory=dict)


class Solver(ABC):
    name: str = "solver"
    exact: bool = False
    max_N: int | None = None

    def applicable(self, N: int) -> bool:
        return self.max_N is None or N <= self.max_N

    @abstractmethod
    def _solve(self, P: PlannerProblem, rng: np.random.Generator) -> tuple[np.ndarray, float, dict]:
        ...

    def solve(self, P: PlannerProblem, rng=None) -> SolveResult:
        rng = np.random.default_rng(rng)
        t0 = time.perf_counter()
        S, evals, info = self._solve(P, rng)
        dt = time.perf_counter() - t0
        S = np.asarray(S, float)
        return SolveResult(S=S, reward=float(P.reward(S)), seconds=dt, evals=float(evals), info=info)

    def describe(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)}


# ================================================================== numba kernels ======
@njit(cache=True)
def _exact_max(W):
    """max_S S^T W S by Gray-code enumeration of 2^(N-1) states (S_{N-1} fixed by symmetry)."""
    N = W.shape[0]
    S = -np.ones(N)
    h = W @ S
    val = S @ h
    best = val
    bestS = S.copy()
    n = 1 << (N - 1)
    for g in range(1, n):
        k = 0
        x = g
        while (x & 1) == 0:
            x >>= 1
            k += 1
        val += -4.0 * S[k] * h[k]
        S[k] = -S[k]
        dk = 2.0 * S[k]
        for i in range(N):
            h[i] += W[i, k] * dk
        if val > best:
            best = val
            for i in range(N):
                bestS[i] = S[i]
    return bestS, best


@njit(cache=True)
def _greedy(W, S):
    """Steepest single-flip ascent of S^T W S until a 1-flip-stable state. Returns S, flips."""
    N = W.shape[0]
    h = W @ S
    flips = 0
    while True:
        kbest = -1
        gbest = 0.0
        for k in range(N):
            g = -4.0 * S[k] * h[k]
            if g > gbest + 1e-12:
                gbest = g
                kbest = k
        if kbest < 0:
            break
        S[kbest] = -S[kbest]
        dk = 2.0 * S[kbest]
        for i in range(N):
            h[i] += W[i, kbest] * dk
        flips += 1
    return S, flips


@njit(cache=True)
def _sa(W, S, betas, sweeps_per_beta, seed):
    np.random.seed(seed)
    N = W.shape[0]
    h = W @ S
    val = S @ h
    best = val
    bestS = S.copy()
    for b in betas:
        for _ in range(sweeps_per_beta):
            for _k in range(N):
                k = np.random.randint(N)
                dH = 2.0 * S[k] * h[k]          # H = -S^T W S / 2
                if dH <= 0.0 or np.random.random() < np.exp(-b * dH):
                    val += -4.0 * S[k] * h[k]
                    S[k] = -S[k]
                    dk = 2.0 * S[k]
                    for i in range(N):
                        h[i] += W[i, k] * dk
                    if val > best:
                        best = val
                        for i in range(N):
                            bestS[i] = S[i]
    return bestS, best


@njit(cache=True)
def _pt(W, S0, betas, n_sweeps, seed):
    np.random.seed(seed)
    R, N = S0.shape
    S = S0.copy()
    H = np.empty((R, N))
    vals = np.empty(R)
    for r in range(R):
        H[r] = W @ S[r]
        vals[r] = S[r] @ H[r]
    best = vals.max()
    bestS = S[np.argmax(vals)].copy()
    perm = np.arange(R)  # perm[k] = replica index at temperature k
    for _ in range(n_sweeps):
        for kt in range(R):
            r = perm[kt]
            b = betas[kt]
            for _k in range(N):
                k = np.random.randint(N)
                dH = 2.0 * S[r, k] * H[r, k]
                if dH <= 0.0 or np.random.random() < np.exp(-b * dH):
                    vals[r] += -4.0 * S[r, k] * H[r, k]
                    S[r, k] = -S[r, k]
                    dk = 2.0 * S[r, k]
                    for i in range(N):
                        H[r, i] += W[i, k] * dk
                    if vals[r] > best:
                        best = vals[r]
                        for i in range(N):
                            bestS[i] = S[r, i]
        for kt in range(R - 1):
            r1, r2 = perm[kt], perm[kt + 1]
            # energies E = -val/2
            d = (betas[kt] - betas[kt + 1]) * (-0.5 * vals[r1] + 0.5 * vals[r2])
            if d >= 0 or np.random.random() < np.exp(d):
                perm[kt], perm[kt + 1] = r2, r1
    return bestS, best


@njit(cache=True)
def _eo(W, S, n_steps, tau, seed):
    np.random.seed(seed)
    N = W.shape[0]
    h = W @ S
    val = S @ h
    best = val
    bestS = S.copy()
    # cumulative distribution P(rank k) ~ k^-tau, k=1..N
    p = np.empty(N)
    for k in range(N):
        p[k] = (k + 1.0) ** (-tau)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    lam = np.empty(N)
    for _ in range(n_steps):
        for i in range(N):
            lam[i] = S[i] * h[i]
        order = np.argsort(lam)          # worst (most frustrated) first
        u = np.random.random()
        r = np.searchsorted(cdf, u)
        k = order[r]
        val += -4.0 * S[k] * h[k]
        S[k] = -S[k]
        dk = 2.0 * S[k]
        for i in range(N):
            h[i] += W[i, k] * dk
        if val > best:
            best = val
            for i in range(N):
                bestS[i] = S[i]
    return bestS, best


def _rand_spins(rng, N, n=None):
    shape = (N,) if n is None else (n, N)
    return np.where(rng.random(shape) < 0.5, 1.0, -1.0)


# ======================================================================= solvers =======
class ExactEnumeration(Solver):
    name = "exact"
    exact = True

    def __init__(self, max_N: int = 30):
        self.max_N = max_N

    def _solve(self, P, rng):
        if P.N > self.max_N:
            raise ValueError("N too large for exact enumeration")
        S, _ = _exact_max(P.W)
        return S, float(2 ** (P.N - 1)), {"certified": True}


class GreedyMultistart(Solver):
    name = "greedy"

    def __init__(self, restarts: int = 50):
        self.restarts = restarts

    def _solve(self, P, rng):
        best, bestS, ev = -np.inf, None, 0
        for _ in range(self.restarts):
            S, fl = _greedy(P.W, _rand_spins(rng, P.N))
            ev += (fl + 1) * P.N
            v = S @ P.W @ S
            if v > best:
                best, bestS = v, S.copy()
        return bestS, ev, {}


class SimulatedAnnealing(Solver):
    name = "sa"

    def __init__(self, sweeps: int = 1000, beta0: float = 0.1, beta1: float = 5.0, restarts: int = 4,
                 n_betas: int = 100):
        self.sweeps, self.beta0, self.beta1, self.restarts, self.n_betas = sweeps, beta0, beta1, restarts, n_betas

    def _solve(self, P, rng):
        betas = np.geomspace(self.beta0, self.beta1, self.n_betas)
        spb = max(1, self.sweeps // self.n_betas)
        best, bestS = -np.inf, None
        for _ in range(self.restarts):
            S, v = _sa(P.W, _rand_spins(rng, P.N), betas, spb, int(rng.integers(2**31)))
            S, _ = _greedy(P.W, S.copy())
            v = S @ P.W @ S
            if v > best:
                best, bestS = v, S.copy()
        return bestS, self.restarts * spb * self.n_betas * P.N, {}


class ParallelTempering(Solver):
    name = "pt"

    def __init__(self, sweeps: int = 2000, n_replicas: int = 16, beta_min: float = 0.2, beta_max: float = 5.0):
        self.sweeps, self.n_replicas, self.beta_min, self.beta_max = sweeps, n_replicas, beta_min, beta_max

    def _solve(self, P, rng):
        betas = np.geomspace(self.beta_min, self.beta_max, self.n_replicas)
        S, _ = _pt(P.W, _rand_spins(rng, P.N, self.n_replicas), betas, self.sweeps, int(rng.integers(2**31)))
        S, _ = _greedy(P.W, S.copy())
        return S, self.sweeps * self.n_replicas * P.N, {}


class ExtremalOptimization(Solver):
    name = "tau_eo"

    def __init__(self, steps_per_spin: int = 200, tau: float = 1.3, restarts: int = 2):
        self.steps_per_spin, self.tau, self.restarts = steps_per_spin, tau, restarts

    def _solve(self, P, rng):
        best, bestS = -np.inf, None
        n = self.steps_per_spin * P.N
        for _ in range(self.restarts):
            S, v = _eo(P.W, _rand_spins(rng, P.N), n, self.tau, int(rng.integers(2**31)))
            S, _ = _greedy(P.W, S.copy())
            v = S @ P.W @ S
            if v > best:
                best, bestS = v, S.copy()
        return bestS, self.restarts * n, {}


class SKGameLearner(Solver):
    """The paper's own learning dynamics used as an optimiser (beta=inf, small alpha),
    run on the symmetric part W. Optional annealed noise: beta(t) from beta0 to infinity."""

    name = "skgame_learning"

    def __init__(self, alpha: float = 0.05, steps: int = 4000, restarts: int = 4, beta0: float | None = None):
        self.alpha, self.steps, self.restarts, self.beta0 = alpha, steps, restarts, beta0

    def _solve(self, P, rng):
        W = P.W
        best, bestS = -np.inf, None
        for _ in range(self.restarts):
            Q = rng.standard_normal(P.N)
            for t in range(self.steps):
                if self.beta0 is None:
                    S = np.where(Q >= 0, 1.0, -1.0)
                else:
                    b = self.beta0 * (1 + 20.0 * t / self.steps) ** 2
                    m = np.tanh(b * Q)
                    S = np.where(rng.random(P.N) < 0.5 * (1 + m), 1.0, -1.0)
                Rf = W @ S
                v = S @ Rf
                if v > best:
                    best, bestS = v, S.copy()
                Q = (1 - self.alpha) * Q + self.alpha * Rf
            S, _ = _greedy(W, np.where(Q >= 0, 1.0, -1.0))
            v = S @ W @ S
            if v > best:
                best, bestS = v, S.copy()
        return bestS, self.restarts * self.steps * P.N, {}


class SDPBurerMonteiro(Solver):
    """Low-rank SDP relaxation max tr(W X), diag X = 1, solved by the mixing method,
    then Goemans-Williamson hyperplane rounding + greedy polish. info['upper_bound'] is the
    SDP value (an upper bound on max S^T W S when the low-rank solution is the SDP optimum)."""

    name = "sdp_gw"

    def __init__(self, rank: int | None = None, iters: int = 300, roundings: int = 200):
        self.rank, self.iters, self.roundings = rank, iters, roundings

    def _solve(self, P, rng):
        N = P.N
        k = self.rank or int(np.ceil(np.sqrt(2 * N))) + 1
        V = rng.standard_normal((N, k))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        W = P.W
        for _ in range(self.iters):
            for i in range(N):
                g = W[i] @ V
                n = np.linalg.norm(g)
                if n > 0:
                    V[i] = g / n
        ub = float(np.einsum("ij,ik,jk->", W, V, V))
        best, bestS = -np.inf, None
        R = rng.standard_normal((k, self.roundings))
        Ss = np.where(V @ R >= 0, 1.0, -1.0).T
        for S in Ss:
            S, _ = _greedy(W, S.copy())
            v = S @ W @ S
            if v > best:
                best, bestS = v, S.copy()
        return bestS, self.iters * N * N / N + self.roundings * N, {"upper_bound_reward": ub / N + P.offset}


class DimodSamplerSolver(Solver):
    """Wrap any dimod sampler (neal, tabu, steepest descent, or a QPU composite)."""

    name = "dimod"

    def __init__(self, sampler_factory: Callable[[], Any], name: str, sample_kwargs: dict | None = None,
                 max_N: int | None = None):
        self._factory = sampler_factory
        self.name = name
        self.sample_kwargs = dict(sample_kwargs or {})
        self.max_N = max_N
        self._sampler = None

    def _solve(self, P, rng):
        if self._sampler is None:
            self._sampler = self._factory()
        bqm = P.to_bqm()
        kw = dict(self.sample_kwargs)
        # only samplers that declare a 'seed' parameter get one: the QPU rejects unknown kwargs
        # server side (not with a TypeError), so trying and catching is not enough there
        params = getattr(self._sampler, "parameters", None)
        if params is None or "seed" in params:
            kw.setdefault("seed", int(rng.integers(2**31)))
        try:
            ss = self._sampler.sample(bqm, **kw)
        except TypeError:
            kw.pop("seed", None)
            ss = self._sampler.sample(bqm, **kw)
        rec = ss.first
        S = np.array([rec.sample[i] for i in range(P.N)], float)
        n_reads = kw.get("num_reads", 1)
        sweeps = kw.get("num_sweeps", 1)
        return S, n_reads * sweeps * P.N, {"energy": float(rec.energy), "info": str(ss.info)[:200]}


def neal_sa(num_reads=10, num_sweeps=1000):
    from dwave.samplers import SimulatedAnnealingSampler
    return DimodSamplerSolver(SimulatedAnnealingSampler, "dwave_sa",
                              {"num_reads": num_reads, "num_sweeps": num_sweeps})


def tabu(num_reads=10, timeout_ms=100):
    from dwave.samplers import TabuSampler
    return DimodSamplerSolver(TabuSampler, "dwave_tabu", {"num_reads": num_reads, "timeout": timeout_ms})


def steepest_descent(num_reads=100):
    from dwave.samplers import SteepestDescentSolver
    return DimodSamplerSolver(SteepestDescentSolver, "dwave_steepest", {"num_reads": num_reads})


def default_solvers(budget: str = "quick") -> list[Solver]:
    q = budget == "quick"
    sv = [ExactEnumeration(max_N=24 if q else 30),
          GreedyMultistart(restarts=20 if q else 200),
          SKGameLearner(alpha=0.05, steps=800 if q else 5000, restarts=2 if q else 8),
          SimulatedAnnealing(sweeps=300 if q else 5000, restarts=2 if q else 8),
          ParallelTempering(sweeps=300 if q else 5000, n_replicas=12 if q else 24),
          ExtremalOptimization(steps_per_spin=60 if q else 500, restarts=1 if q else 4),
          SDPBurerMonteiro(iters=60 if q else 400, roundings=50 if q else 500)]
    try:
        sv += [neal_sa(num_reads=4 if q else 50, num_sweeps=300 if q else 5000),
               tabu(num_reads=2 if q else 20, timeout_ms=20 if q else 500)]
    except ImportError:  # pragma: no cover
        pass
    return sv
