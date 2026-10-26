"""Counting fixed points and limit cycles (Sec. IV, V, Appendix B).

* count_fixed_points_T0     : exact Gray-code enumeration of S_i = sign(sum_j J_ij S_j)
* count_two_cycles_T0       : exact count of 2-cycles of the alpha=1 parallel map (via pairs)
* count_nmfe_solutions      : multi-start Newton count of m = tanh(beta J m) (finite T, lower bound)
* annealed_complexity       : (1/N) log mean_J N_J
* Sigma2_saddle             : L=2 cycle complexity with memory alpha (Eq. B40), symmetric ansatz
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import optimize, special
from scipy.stats import norm

from ..model import InteractionEnsemble, eta as eta_of_eps
from .analytic import complexity_fp_T0


# ------------------------------------------------------------- exact enumeration, T=0 --
@njit(cache=True)
def _count_fp_gray(J):
    """Number of S in {-1,1}^N with S_i h_i > 0 for all i, h = J S (ties h_i=0 excluded)."""
    N = J.shape[0]
    S = -np.ones(N)
    h = J @ S
    count = 0
    ok = True
    for i in range(N):
        if S[i] * h[i] <= 0:
            ok = False
            break
    if ok:
        count += 1
    n_states = 1 << N
    for g in range(1, n_states):
        # bit to flip in Gray code = index of lowest set bit of g
        k = 0
        x = g
        while (x & 1) == 0:
            x >>= 1
            k += 1
        S[k] = -S[k]
        dk = 2.0 * S[k]
        for i in range(N):
            h[i] += J[i, k] * dk
        ok = True
        for i in range(N):
            if S[i] * h[i] <= 0:
                ok = False
                break
        if ok:
            count += 1
    return count


def count_fixed_points_T0(J: np.ndarray) -> int:
    J = np.ascontiguousarray(J, dtype=np.float64)
    if J.shape[0] > 34:
        raise ValueError("exact enumeration limited to N <= 34")
    return int(_count_fp_gray(J))


@njit(cache=True)
def _count_two_cycles_gray(J):
    """Number of ordered pairs (S, S') with S' = sign(J S), S = sign(J S'), S' != S.

    Enumerate S, compute S' = sign(JS) (requires no ties), test sign(J S') == S.
    Each genuine 2-cycle {S, S'} is counted twice (once from each element).
    """
    N = J.shape[0]
    S = -np.ones(N)
    h = J @ S
    Sp = np.empty(N)
    count = 0
    n_states = 1 << N
    for g in range(n_states):
        if g > 0:
            k = 0
            x = g
            while (x & 1) == 0:
                x >>= 1
                k += 1
            S[k] = -S[k]
            dk = 2.0 * S[k]
            for i in range(N):
                h[i] += J[i, k] * dk
        tie = False
        same = True
        for i in range(N):
            if h[i] == 0.0:
                tie = True
            Sp[i] = 1.0 if h[i] > 0 else -1.0
            if Sp[i] != S[i]:
                same = False
        if tie or same:
            continue
        ok = True
        for i in range(N):
            hp = 0.0
            for j in range(N):
                hp += J[i, j] * Sp[j]
            if hp * S[i] <= 0:
                ok = False
                break
        if ok:
            count += 1
    return count


def count_two_cycles_T0(J: np.ndarray) -> int:
    return int(_count_two_cycles_gray(np.ascontiguousarray(J, dtype=np.float64)) // 2)


def annealed_complexity(counts: np.ndarray, N: int) -> float:
    counts = np.asarray(counts, float)
    mean = counts.mean()
    return float(np.log(mean) / N) if mean > 0 else -np.inf


def quenched_complexity(counts: np.ndarray, N: int) -> float:
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    return float(np.mean(np.log(counts)) / N) if len(counts) else -np.inf


# --------------------------------------------- finite-T NMFE solutions, multi-start --
def count_nmfe_solutions(J: np.ndarray, beta: float, n_starts: int = 400, rng=None,
                         tol: float = 1e-9, merge: float = 1e-4) -> tuple[int, np.ndarray]:
    """Lower bound on the number of solutions of m = tanh(beta J m)   (Eq. 19).

    Newton (scipy 'hybr') from random starts m0 in [-1,1]^N; distinct solutions are merged
    when their sup-distance is < merge. The paramagnet m=0 is always included if stable or not.
    """
    rng = np.random.default_rng(rng)
    N = J.shape[0]
    sols = []

    def F(m):
        return m - np.tanh(beta * (J @ m))

    def Jac(m):
        th = np.tanh(beta * (J @ m))
        return np.eye(N) - beta * (1 - th * th)[:, None] * J

    for _ in range(n_starts):
        m0 = rng.uniform(-1, 1, N)
        # a few damped iterations first (helps symmetric cases), then Newton
        m = m0
        for _k in range(20):
            m = 0.5 * m + 0.5 * np.tanh(beta * (J @ m))
        r = optimize.root(F, m, jac=Jac, method="hybr", tol=tol)
        if not r.success or np.max(np.abs(F(r.x))) > 1e-7:
            continue
        x = r.x
        if not any(np.max(np.abs(x - y)) < merge for y in sols):
            sols.append(x)
    if not any(np.max(np.abs(y)) < merge for y in sols):
        sols.append(np.zeros(N))
    return len(sols), np.array(sols)


# --------------------------------------------------- L=2 complexity with memory (B40) --
def bvn_cdf_diag(z, rho):
    """Phi_2(z, z; rho) = Phi(z) - 2 T(z, sqrt((1-rho)/(1+rho)))  (Owen 1956), smooth in z, rho."""
    z = np.asarray(z, float)
    rho = np.clip(np.asarray(rho, float), -1 + 1e-14, 1 - 1e-14)
    return norm.cdf(z) - 2.0 * special.owens_t(z, np.sqrt((1 - rho) / (1 + rho)))


def _sigma2_F(x, alpha, et):
    """F(r, k, v, w) of Eq. (B40) under the 2-cycle symmetric ansatz
    V(1,1)=V(2,2)=v, V(1,2)=V(2,1)=w, r = iR_hat(1,2), k = iK_hat(1,2)."""
    r, k, v, w = x
    a = 1.0 - alpha
    D = 1.0 - a * a
    tot = 0.0
    for S in (+1.0, -1.0):
        G = et / D * ((w + a * v) + S * (v + a * w))     # Gamma_1 = Gamma_2 under the ansatz
        c11 = (1 + a * a - 2 * a * r) / D**2
        c12 = S * (2 * a - (1 + a * a) * r) / D**2
        rho = np.clip(c12 / c11, -1 + 1e-12, 1 - 1e-12)
        z = G / np.sqrt(c11)
        tot += np.exp(S * k) * float(bvn_cdf_diag(z, rho))
    return r * k - et * (v * v + w * w) + np.log(2.0) + np.log(max(tot, 1e-300))


def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def Sigma2_saddle(alpha: float, et: float, x0=None):
    """Return (Sigma_2, x*=(r,k,v,w)) at the saddle of Eq. (B40) (2-cycle ansatz).

    At alpha=1 the saddle is r=k=v=0, w=x_FP and Sigma_2 = 2 Sigma_FP(eta) (Eq. B50).
    Use continuation in alpha for alpha<1 (Fig. 10). 1 - C(tau=1) = 1 + r.
    """
    if x0 is None:
        _, xfp = complexity_fp_T0(et)
        x0 = np.array([0.0, 0.0, 0.0, xfp])
    f = lambda x: _sigma2_F(x, alpha, et)
    g = lambda x: _num_grad(f, x)
    x0 = np.asarray(x0, float)
    best = None
    for method in ("hybr", "lm"):
        sol = optimize.root(g, x0, method=method, tol=1e-11)
        res = np.max(np.abs(g(sol.x)))
        if res < 1e-6 and np.all(np.isfinite(sol.x)):
            return float(f(sol.x)), sol.x, True
        if best is None or res < best[1]:
            best = (sol.x, res)
    return float(f(best[0])), best[0], False


def Sigma2_curve(alphas, et):
    alphas = np.sort(np.asarray(alphas))[::-1]  # from alpha=1 downward
    x = None
    rows = []
    for a in alphas:
        S, x_new, ok = Sigma2_saddle(float(a), et, x0=x)
        if ok and x is not None and np.max(np.abs(x_new - x)) > 0.5:
            ok = False          # jumped to another saddle: reject, keep continuation on the branch
        if ok:
            x = x_new
            rows.append((a, S, 1.0 + x_new[0], 1))
        else:
            rows.append((a, np.nan, np.nan, 0))
    return np.array(rows)  # columns: alpha, Sigma2, 1-C(1), converged
