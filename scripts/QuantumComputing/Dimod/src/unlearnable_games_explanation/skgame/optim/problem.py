"""The social planner's problem: maximise the average reward R_N(S) over S in {-1,1}^N.

R_N(S) = (1/N) S^T J S = (1/N) S^T W S,  W = (J + J^T)/2 with zero diagonal (diag terms are
constants since S_i^2 = 1). Equivalently minimise the SK energy

    H(S) = - sum_{i<j} W_ij S_i S_j ,       R_N = -2 H / N  (+ trace(J)/N).

For eps = 0 this is exactly the SK ground-state problem (NP-hard in the worst case; for
typical instances the best known exact methods are exponential, cf. footnote [58] for the
polynomial-time 1-eps approximation of Montanari).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlannerProblem:
    J: np.ndarray
    name: str = ""

    def __post_init__(self):
        J = np.asarray(self.J, float)
        self.N = J.shape[0]
        W = 0.5 * (J + J.T)
        self.offset = float(np.trace(W)) / self.N
        np.fill_diagonal(W, 0.0)
        self.W = np.ascontiguousarray(W)

    def reward(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, float)
        if S.ndim == 1:
            return float(S @ self.W @ S) / self.N + self.offset
        return np.einsum("ki,ij,kj->k", S, self.W, S) / self.N + self.offset

    def energy(self, S: np.ndarray):
        """SK energy H = -sum_{i<j} W_ij S_i S_j."""
        return -0.5 * (self.reward(S) - self.offset) * self.N

    def reward_from_energy(self, H):
        return -2.0 * np.asarray(H) / self.N + self.offset

    def to_bqm(self):
        """dimod BinaryQuadraticModel (SPIN) with E = H(S). Lazy import."""
        import dimod
        iu = np.triu_indices(self.N, 1)
        quad = {(int(i), int(j)): -float(self.W[i, j]) for i, j in zip(*iu)}
        return dimod.BinaryQuadraticModel({}, quad, 0.0, dimod.SPIN)

    def to_ising(self):
        """(h, J) dictionaries for D-Wave style samplers."""
        iu = np.triu_indices(self.N, 1)
        return {}, {(int(i), int(j)): -float(self.W[i, j]) for i, j in zip(*iu)}

    @classmethod
    def random(cls, N: int, eps: float = 0.0, rng=None):
        from ..model import InteractionEnsemble
        rng = np.random.default_rng(rng)
        return cls(InteractionEnsemble(N, eps).sample(rng, 1)[0])
