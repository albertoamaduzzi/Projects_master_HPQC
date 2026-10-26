"""Discrete-time Dynamical Mean-Field Theory of the SK-game (Sec. VI, Appendix C).

Effective single-agent process (N -> infinity), discrete time, convention of this package:

    Q(t+1) = (1-alpha) Q(t) + alpha [ (1-eps) sum_{s<t} G(t,s) m(s) + phi(t) + h(t) ]
    m(t)   = tanh(beta Q(t))           (sign(Q) at beta = inf)
    <phi(t) phi(s)> = ups(eps) [ C(t,s) + (1 - q(t)) delta_{ts} ]   (online learning, Eq. 31)
    C(t,s) = <m(t) m(s)>,  q(t) = C(t,t),  G(t,s) = < d m(t) / d h(s) >

(The paper's Eq. (30) writes alpha^2 (1-eps) G with G defined w.r.t. a field entering without
the factor alpha; Appendix C notes the rescaling G -> alpha G.  Both are the same process.)

Self-consistency is solved by Monte Carlo sampling of M paths and damped iteration
(Eissfeller & Opper 1992). The response is obtained without finite differences through
Novikov's theorem: since phi and h enter identically,
    <m(t) phi(s)> = sum_{u<t} K(s,u) G(t,u)  =>  G(t, :t) = K[:t,:t]^{-1} <m(t) phi(:t)>.
The Cholesky factor of a leading block of K is the leading block of chol(K), so all the
causal solves cost O(T^3) in total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from ..model import upsilon


@dataclass
class DMFTResult:
    C: np.ndarray            # (T, T)
    G: np.ndarray            # (T, T), strictly lower triangular
    q: np.ndarray            # (T,)
    history: list = field(default_factory=list)  # convergence ||C_new - C||_max per iteration
    m_paths: np.ndarray | None = None

    def stationary_C(self, max_lag: int, window: int) -> np.ndarray:
        T = self.C.shape[0]
        t0 = max(T - window, 0)
        out = np.empty(max_lag + 1)
        for tau in range(max_lag + 1):
            ts = np.arange(t0, T - tau)
            out[tau] = np.mean(self.C[ts, ts + tau]) if len(ts) else np.nan
        return out


@dataclass
class DMFTSolver:
    alpha: float
    beta: float
    eps: float
    T: int = 200
    M: int = 20000
    n_iter: int = 30
    damping: float = 0.5
    ridge: float = 1e-6
    online: bool = True
    q0_scale: float = 1.0
    seed: int = 0
    use_response: bool = True

    def _m(self, Q):
        if not np.isfinite(self.beta):
            return np.where(Q >= 0, 1.0, -1.0)
        return np.tanh(self.beta * Q)

    def _sample_phi(self, K, rng):
        w, V = linalg.eigh(K)
        w = np.clip(w, 0.0, None)
        Z = rng.standard_normal((self.M, K.shape[0]))
        return (Z * np.sqrt(w)) @ V.T

    def _simulate(self, phi, G, rng):
        M, T = phi.shape
        a = self.alpha
        Q = self.q0_scale * rng.standard_normal(M)
        m = np.empty((M, T))
        for t in range(T):
            m[:, t] = self._m(Q)
            mem = (1.0 - self.eps) * (m[:, :t] @ G[t, :t]) if (t > 0 and self.use_response) else 0.0
            Q = (1.0 - a) * Q + a * (mem + phi[:, t])
        return m

    def _response(self, m, phi, K):
        M, T = m.shape
        B = (m.T @ phi) / M           # B[t, s] = <m(t) phi(s)>
        Kr = K + self.ridge * np.trace(K) / T * np.eye(T)
        L = linalg.cholesky(Kr, lower=True)
        G = np.zeros((T, T))
        for t in range(1, T):
            Lt = L[:t, :t]
            y = linalg.solve_triangular(Lt, B[t, :t], lower=True)
            G[t, :t] = linalg.solve_triangular(Lt.T, y, lower=False)
        return G

    def _K(self, C):
        u = upsilon(self.eps)
        q = np.diag(C)
        K = u * C.copy()
        if self.online:
            K[np.diag_indices_from(K)] += u * (1.0 - q)
        return 0.5 * (K + K.T)

    def solve(self, C_init: np.ndarray | None = None, keep_paths: int = 0, verbose: bool = False) -> DMFTResult:
        rng = np.random.default_rng(self.seed)
        T = self.T
        C = np.eye(T) if C_init is None else C_init.copy()
        G = np.zeros((T, T))
        hist = []
        need_G = self.use_response and abs(1.0 - self.eps) > 1e-12
        for it in range(self.n_iter):
            K = self._K(C)
            phi = self._sample_phi(K, rng)
            m = self._simulate(phi, G, rng)
            C_new = (m.T @ m) / self.M
            if need_G:
                G_new = self._response(m, phi, K)
                G = self.damping * G + (1 - self.damping) * G_new
            diff = float(np.max(np.abs(C_new - C)))
            C = self.damping * C + (1 - self.damping) * C_new
            hist.append(diff)
            if verbose:
                print(f"DMFT iter {it}: max|dC| = {diff:.3e}")
        return DMFTResult(C=C, G=G, q=np.diag(C).copy(), history=hist,
                          m_paths=m[:keep_paths] if keep_paths else None)


def continuous_eps1_alpha0(tau_max=12.0, n=1201):
    """Continuous-time eps=1, alpha->0 limit: Q' = -Q + phi, <phi phi> = C/2  (Eq. 36-39)."""
    from .analytic import sompolinsky_crisanti
    return sompolinsky_crisanti(tau_max, n)
