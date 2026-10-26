"""Observables used throughout the paper (pure functions of recorded arrays)."""
from __future__ import annotations

import numpy as np


def average_reward(S: np.ndarray, J: np.ndarray) -> np.ndarray:
    """R_N = (1/N) S^T J S   (Eq. 12). S: (R,N), J: (R,N,N) -> (R,)."""
    return np.einsum("ri,rij,rj->r", S, J, S) / S.shape[-1]


def individual_rewards(S: np.ndarray, J: np.ndarray) -> np.ndarray:
    """e_i = S_i sum_j J_ij S_j   (Eq. 16)."""
    return S * np.einsum("rij,rj->ri", J, S)


def stationary_autocorr(X: np.ndarray, max_lag: int) -> np.ndarray:
    """C(tau) = < (1/N) sum_i X_i(t) X_i(t+tau) >_{t, realisations}   (Eq. 15, TTI form).

    X: (R, T, N). Returns per-realisation curves of shape (R, max_lag+1).
    """
    X = X.astype(np.float64)
    R, T, N = X.shape
    max_lag = min(max_lag, T - 1)
    out = np.empty((R, max_lag + 1))
    for tau in range(max_lag + 1):
        out[:, tau] = np.mean(np.sum(X[:, : T - tau] * X[:, tau:], axis=2) / N, axis=1)
    return out


def q_param(m: np.ndarray) -> np.ndarray:
    """q = (1/N) sum_i m_i^2 (equal-time autocorrelation C(0))."""
    return np.mean(m.astype(np.float64) ** 2, axis=-1)


def magnetisation(X: np.ndarray) -> np.ndarray:
    """M = (1/N) sum_j X_j   (Eq. 11)."""
    return np.mean(X, axis=-1)


def cross_overlap(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """C^x_N = (1/N) sum_i (e^a_i - <e^a>)(e^b_i - <e^b>)   (Eq. 17), batched on leading axes."""
    da = ea - ea.mean(-1, keepdims=True)
    db = eb - eb.mean(-1, keepdims=True)
    return np.mean(da * db, axis=-1)


def frozen_fraction(S: np.ndarray) -> np.ndarray:
    """Fraction of agents with S_i(t+1) = S_i(t), averaged over the window. S: (R,T,N)."""
    return np.mean(S[:, 1:] == S[:, :-1], axis=(1, 2))


def detect_period(C: np.ndarray, tol: float = 1e-9, max_L: int | None = None) -> int:
    """Smallest L>=1 with C(nL) = 1 for every multiple available (exact cycle); 0 if none."""
    n = len(C)
    max_L = max_L or n - 1
    for L in range(1, max_L + 1):
        idx = np.arange(L, n, L)
        if len(idx) and np.all(np.abs(C[idx] - 1.0) < tol):
            return L
    return 0


def classify_phase(S_window: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Per realisation: 1 = fixed point, 2 / 4 / L = exact cycle of length L, 0 = no recurrence."""
    C = stationary_autocorr(S_window, max_lag=min(16, S_window.shape[1] - 1))
    return np.array([detect_period(c, tol) for c in C])


def dominant_frequency(C: np.ndarray, dt: float = 1.0, angular: bool = False, pad: int = 8) -> float:
    """Frequency of the largest non-zero peak of |FFT(C - mean C)|^2 (zero-padded).

    Default: ordinary frequency in cycles per time step (the convention matching Fig. 15,
    where the fundamental and its 3rd harmonic both fall in omega/sqrt(alpha) < 1).
    """
    x = C - C.mean()
    n = pad * len(x)
    P = np.abs(np.fft.rfft(x, n=n)) ** 2
    f = np.fft.rfftfreq(n, d=dt)
    k = 1 + int(np.argmax(P[1:]))
    return float(2 * np.pi * f[k] if angular else f[k])


def survival(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.sort(np.ravel(x))
    return 1.0 - np.searchsorted(x, grid, side="right") / len(x)
