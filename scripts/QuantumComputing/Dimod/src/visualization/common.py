"""Shared helpers for the SK_game and optimal_algorithms visualizations.

The vocabulary (defined in SK_game/README.md, section "Glossary") is the same in both folders and
in the code below:

  spin / decision      S_i = +-1
  couplings            J_ij (game), W_ij = (J_ij + J_ji)/2 (planner, zero diagonal)
  local field          h_i = sum_j W_ij S_j
  satisfaction         lambda_i = S_i h_i           (> 0: agent i is happy with its decision)
  energy               H(S) = -sum_{i<j} W_ij S_i S_j = -(1/2) sum_i lambda_i
  reward               R_N(S) = (1/N) S^T J S = -2 H / N  (zero diagonal)
  single-flip move     S_k -> -S_k, energy change dH_k = 2 lambda_k
  local minimum        no single flip lowers H  <=>  lambda_i >= 0 for all i  (1-flip stable)
  ground state         global minimum of H = maximum of R_N
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "unlearnable_games_explanation"
sys.path.insert(0, str(PKG))

from skgame.model import InteractionEnsemble  # noqa: E402
from skgame.optim.problem import PlannerProblem  # noqa: E402

UP, DOWN = "#d62728", "#1f77b4"          # S_i = +1 / -1
POS, NEG = "#2ca02c", "#9467bd"          # W_ij > 0 (wants agreement) / < 0 (wants disagreement)


def instance(N: int, eps: float = 0.0, J0: float = 0.0, seed: int = 0) -> tuple[np.ndarray, PlannerProblem]:
    """(J, planner problem) for one draw of the SK-game ensemble."""
    J = InteractionEnsemble(N, eps, J0=J0).sample(np.random.default_rng(seed), 1)[0]
    return J, PlannerProblem(J)


def local_field(W, S):
    return W @ S


def satisfaction(W, S):
    return S * (W @ S)


def energy(W, S):
    S = np.asarray(S, float)
    if S.ndim == 1:
        return -0.5 * S @ W @ S
    return -0.5 * np.einsum("ki,ij,kj->k", S, W, S)


def all_states(N: int) -> np.ndarray:
    """Every configuration of N spins, (2^N, N), state index = binary code (bit 1 -> S_i = -1)."""
    return 1.0 - 2.0 * np.array(list(itertools.product([0, 1], repeat=N)), float)


def landscape(W) -> dict:
    """Exhaustive energy landscape of a small instance: energies, local minima, basins of the
    steepest-descent dynamics, ground state. N <= ~16."""
    N = W.shape[0]
    S = all_states(N)
    E = energy(W, S)
    lam = S * (S @ W)                                # satisfaction of every spin in every state
    is_min = np.all(lam >= -1e-12, axis=1)
    idx = {tuple(s): k for k, s in enumerate(S.astype(int))}
    # steepest descent: flip the spin with the most negative satisfaction until none is negative
    nxt = np.arange(len(S))
    for k in range(len(S)):
        j = np.argmin(lam[k])
        if lam[k, j] < -1e-12:
            s = S[k].copy()
            s[j] = -s[j]
            nxt[k] = idx[tuple(s.astype(int))]
    basin = nxt.copy()
    for _ in range(N + 1):
        basin = nxt[basin]
    g = int(np.argmin(E))
    return {"S": S, "E": E, "is_min": is_min, "basin": basin, "ground": g}


def hamming_to(S, ref):
    """Distance to a reference state modulo the global flip S -> -S (same energy)."""
    d = np.sum(S != ref, axis=-1)
    return np.minimum(d, S.shape[-1] - d)


def save(fig, folder: Path, name: str) -> Path:
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    p = folder / "figures" / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def suptitle(fig, text):
    fig.suptitle(text, y=1.03, fontsize=12)
