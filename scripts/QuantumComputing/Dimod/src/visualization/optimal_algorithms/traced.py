"""Traced versions of the optimisers of skgame/optim/solvers.py, for pictures.

Same update rules as the numba kernels used in the benchmark, written in plain NumPy and
recording what happens at every step (energy, which spin moved, acceptance, temperatures...).
Vocabulary as in ../SK_game/README.md: spins S, symmetric couplings W, local fields h = W S,
satisfactions lambda = S * h, energy H = -S W S / 2, single-flip change dH_k = 2 lambda_k.
Only meant for small N.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply


def energy(W, S):
    return -0.5 * S @ W @ S


def flip(W, S, h, k):
    """Single-flip move on S (in place) with the O(N) local-field update; returns dH."""
    dH = 2.0 * S[k] * h[k]
    S[k] = -S[k]
    h += 2.0 * W[:, k] * S[k]
    return dH


# ------------------------------------------------------------------ exact (Gray code) ----
def gray_code(W, n_steps=None):
    """Visit configurations in Gray-code order (one flip per step, S_{N-1} fixed by the global
    flip symmetry). Returns the flipped spin and the energy at every step, and the best state."""
    N = W.shape[0]
    S = -np.ones(N)
    h = W @ S
    E = energy(W, S)
    n = 1 << (N - 1)
    n_steps = n - 1 if n_steps is None else min(n_steps, n - 1)
    ks, Es = [], [E]
    best, bestS = E, S.copy()
    for g in range(1, n_steps + 1):
        k = (g & -g).bit_length() - 1             # index of the lowest set bit of g
        E += flip(W, S, h, k)
        ks.append(k)
        Es.append(E)
        if E < best:
            best, bestS = E, S.copy()
    return {"k": np.array(ks), "E": np.array(Es), "best": best, "bestS": bestS}


# --------------------------------------------------------------------- steepest descent --
def steepest_descent(W, S0):
    """Flip the most unsatisfied spin (most negative lambda) until every lambda >= 0."""
    S = S0.copy()
    h = W @ S
    Es = [energy(W, S)]
    while True:
        lam = S * h
        k = int(np.argmin(lam))
        if lam[k] >= -1e-12:
            break
        Es.append(Es[-1] + flip(W, S, h, k))
    return {"E": np.array(Es), "S": S}


# ------------------------------------------------------------------ simulated annealing --
def simulated_annealing(W, S0, betas, sweeps_per_beta, rng):
    """Metropolis single flips: pick k at random, accept with min(1, exp(-beta dH_k)); one
    sweep = N attempts; beta follows the schedule `betas` (cold at the end)."""
    N = W.shape[0]
    S = S0.copy()
    h = W @ S
    E = energy(W, S)
    Es, acc = [E], []
    best, bestS = E, S.copy()
    for b in betas:
        a = 0
        for _ in range(sweeps_per_beta):
            for _ in range(N):
                k = rng.integers(N)
                dH = 2.0 * S[k] * h[k]
                if dH <= 0 or rng.random() < np.exp(-b * dH):
                    E += flip(W, S, h, k)
                    a += 1
                    if E < best:
                        best, bestS = E, S.copy()
            Es.append(E)
        acc.append(a / (sweeps_per_beta * N))
    # like skgame's SimulatedAnnealing: keep the best state seen, then relax it by steepest descent
    polished = steepest_descent(W, bestS)
    return {"E": np.array(Es), "acceptance": np.array(acc), "S": S, "best": best,
            "best_polished": polished["E"][-1]}


# ------------------------------------------------------------------ parallel tempering ---
def parallel_tempering(W, betas, n_sweeps, rng):
    """One replica per temperature; Metropolis sweeps on each, then swap neighbouring
    temperatures with probability min(1, exp((beta_k - beta_{k+1}) (E_k - E_{k+1})))."""
    R, N = len(betas), W.shape[0]
    S = np.where(rng.random((R, N)) < 0.5, 1.0, -1.0)
    h = S @ W
    E = np.array([energy(W, s) for s in S])
    perm = np.arange(R)                           # perm[t] = replica currently at temperature t
    E_hist, where = [], []                        # energy at each temperature; temperature of each replica
    swaps_try, swaps_acc = np.zeros(R - 1), np.zeros(R - 1)
    best = E.min()
    best_hist = []
    for _ in range(n_sweeps):
        for t in range(R):
            r = perm[t]
            for _ in range(N):
                k = rng.integers(N)
                dH = 2.0 * S[r, k] * h[r, k]
                if dH <= 0 or rng.random() < np.exp(-betas[t] * dH):
                    E[r] += flip(W, S[r], h[r], k)
        for t in range(R - 1):
            r1, r2 = perm[t], perm[t + 1]
            swaps_try[t] += 1
            d = (betas[t] - betas[t + 1]) * (E[r1] - E[r2])
            if d >= 0 or rng.random() < np.exp(d):
                perm[t], perm[t + 1] = r2, r1
                swaps_acc[t] += 1
        best = min(best, E.min())
        best_hist.append(best)
        E_hist.append(E[perm].copy())
        inv = np.empty(R, int)
        inv[perm] = np.arange(R)
        where.append(inv.copy())
    return {"E": np.array(E_hist), "temperature_of_replica": np.array(where),
            "swap_rate": swaps_acc / np.maximum(swaps_try, 1), "best": np.array(best_hist)}


# --------------------------------------------------------------- extremal optimisation ---
def tau_eo(W, S0, n_steps, tau, rng):
    """Rank the spins from most to least unsatisfied (lambda ascending); flip the spin of rank k
    with probability ~ k^-tau, ALWAYS (no acceptance test). Keeps the best state seen."""
    N = W.shape[0]
    S = S0.copy()
    h = W @ S
    E = energy(W, S)
    p = np.arange(1, N + 1, dtype=float) ** (-tau)
    cdf = np.cumsum(p) / p.sum()
    Es, ranks = [E], []
    best = E
    for _ in range(n_steps):
        order = np.argsort(S * h)
        r = int(np.searchsorted(cdf, rng.random()))
        E += flip(W, S, h, order[r])
        ranks.append(r + 1)
        Es.append(E)
        best = min(best, E)
    return {"E": np.array(Es), "rank": np.array(ranks), "best": best}


# ------------------------------------------------------------------------- tabu search ---
def tabu_search(W, S0, n_steps, tenure):
    """Each step flips the best spin that is not tabu (lowest dH = 2 lambda_k, even if dH > 0);
    a flipped spin stays tabu for `tenure` steps (aspiration: allowed if it beats the best)."""
    N = W.shape[0]
    S = S0.copy()
    h = W @ S
    E = energy(W, S)
    best = E
    until = np.zeros(N, int)
    Es, moved = [E], []
    for t in range(1, n_steps + 1):
        dH = 2.0 * S * h
        ok = (until < t) | (E + dH < best - 1e-12)
        dH_masked = np.where(ok, dH, np.inf)
        k = int(np.argmin(dH_masked))
        E += flip(W, S, h, k)
        until[k] = t + tenure
        best = min(best, E)
        Es.append(E)
        moved.append(k)
    return {"E": np.array(Es), "moved": np.array(moved), "best": best}


# ----------------------------------------------------------------------- game learning ---
def game_learning(W, alpha, steps, rng, beta0=None):
    """The SK game as an optimiser (SKGameLearner): S = sign Q (or noisy with annealed beta),
    Q <- (1-alpha) Q + alpha h, with h = W S the local fields (= rewards R_i at eps = 0)."""
    N = W.shape[0]
    Q = rng.standard_normal(N)
    Es, flips = [], []
    S_prev = None
    for t in range(steps):
        if beta0 is None:
            S = np.where(Q >= 0, 1.0, -1.0)
        else:
            b = beta0 * (1 + 20.0 * t / steps) ** 2
            S = np.where(rng.random(N) < 0.5 * (1 + np.tanh(b * Q)), 1.0, -1.0)
        h = W @ S
        Es.append(-0.5 * S @ h)
        flips.append(0 if S_prev is None else int(np.sum(S != S_prev)))
        S_prev = S
        Q = (1 - alpha) * Q + alpha * h
    return {"E": np.array(Es), "flips": np.array(flips), "S": S, "lambda": S * (W @ S)}


# ------------------------------------------------------------------ SDP + Goemans-Williamson
def sdp_mixing(W, k, iters, rng):
    """Relax S_i in {-1,1} to unit vectors v_i in R^k and maximise sum_ij W_ij v_i.v_j by the
    mixing method: v_i <- normalise(sum_j W_ij v_j), one spin at a time."""
    N = W.shape[0]
    V = rng.standard_normal((N, k))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    obj = []
    for _ in range(iters):
        for i in range(N):
            g = W[i] @ V
            n = np.linalg.norm(g)
            if n > 0:
                V[i] = g / n
        obj.append(-0.5 * np.einsum("ij,ik,jk->", W, V, V))      # relaxed energy (lower bound on H)
    return {"V": V, "relaxed_energy": np.array(obj)}


def gw_rounding(W, V, n, rng):
    """Random hyperplane r: S_i = sign(v_i . r). Returns the energies of n roundings."""
    R = rng.standard_normal((V.shape[1], n))
    S = np.where(V @ R >= 0, 1.0, -1.0).T
    return S, -0.5 * np.einsum("ki,ij,kj->k", S, W, S)


# --------------------------------------------------------------------- quantum annealing -
def _pauli_x_sum(N):
    """sum_i sigma^x_i as a sparse 2^N x 2^N matrix."""
    from scipy.sparse import csr_matrix
    dim = 1 << N
    rows, cols = [], []
    idx = np.arange(dim)
    for i in range(N):
        rows.append(idx)
        cols.append(idx ^ (1 << i))
    return csr_matrix((np.ones(N * dim), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))


def diag_energies(W):
    """Classical energy H(S) of every basis state |z>, bit i = 1 <-> S_i = -1."""
    N = W.shape[0]
    z = np.arange(1 << N)
    S = 1.0 - 2.0 * ((z[:, None] >> np.arange(N)) & 1)
    return -0.5 * np.einsum("ki,ij,kj->k", S, W, S), S


def qa_hamiltonian(W, s, scale):
    """H(s) = -A(s) sum_i sigma^x_i + B(s) H_P / scale, with the idealised schedule A = 1-s, B = s
    (D-Wave's A(s), B(s) have the same roles and different shapes)."""
    from scipy.sparse import diags
    Ez, _ = diag_energies(W)
    return -(1 - s) * _pauli_x_sum(W.shape[0]) + s * diags(Ez / scale)


def qa_spectrum(W, s_grid, n_levels=8):
    """Lowest levels of H(s) along the anneal (dense diagonalisation, N <= 10)."""
    scale = np.abs(W).max()
    out = []
    for s in s_grid:
        ev = np.linalg.eigvalsh(qa_hamiltonian(W, s, scale).toarray())
        out.append(ev[:n_levels])
    return np.array(out)


def qa_evolve(W, T, n_steps=400):
    """Schrodinger evolution |psi(0)> = |+>^N (ground state of the transverse field) under H(t/T)
    for total time T (units hbar / energy scale). Returns the final probabilities of every
    classical configuration."""
    N = W.shape[0]
    scale = np.abs(W).max()
    psi = np.full(1 << N, 1 / np.sqrt(1 << N), dtype=complex)
    dt = T / n_steps
    for m in range(n_steps):
        s = (m + 0.5) / n_steps
        psi = expm_multiply(-1j * dt * qa_hamiltonian(W, s, scale), psi)
    return np.abs(psi) ** 2
