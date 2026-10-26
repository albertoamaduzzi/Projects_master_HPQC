"""Parameter points ("regimes") where the planner's optimum is easy or hard to retrieve.

Two different questions share the same instance J:

1. Static: how hard is max_S S^T W S for an optimiser (annealer, SA, PT, ...)?
   Off the diagonal W_ij = (1-eps/2) J^S_ij + J0/N, i.e. an SK model with coupling scale
   sigma_eff = 1-eps/2 and mean J0, so only the ratio

       J0_eff = J0 / (1 - eps/2)

   matters (eps alone only rescales the landscape). At T=0 the SK model is a spin glass for
   J0_eff < 1 (exponentially many metastable states, NP-hard in the worst case) and
   ferromagnetic for J0_eff > 1 (the optimum is close to the aligned state S_i = +-1 for all i,
   found by greedy descent). J0_eff = 1 is the boundary.

2. Dynamical: does the game itself (the learning map with alpha, beta, eps) reach it?
   At beta = inf and eps < eps_c ~ 0.8 the dynamics freezes on fixed points ("satisficing",
   reward below the optimum); at alpha = 1 it locks on 2-cycles; for eps > eps_c it is
   chaotic; for T = 1/beta > T_c(eps) it is paramagnetic (m = 0, reward ~ 0).

Each `Regime` fixes (eps, J0) for the instance and (alpha, beta) for the game. Regimes with
the same (eps, J0) share instances, so the annealer is called once per problem class.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from ..theory.analytic import Tc, eps_c_estimate

INF = float("inf")


def j0_eff(eps: float, J0: float) -> float:
    return J0 / (1.0 - eps / 2.0) if eps < 2 else INF


def planner_phase(eps: float, J0: float, width: float = 0.25) -> str:
    """T=0 phase of the planner's problem: 'spin_glass', 'boundary' or 'ferromagnetic'."""
    j = j0_eff(eps, J0)
    if abs(j - 1.0) <= width:
        return "boundary"
    return "ferromagnetic" if j > 1.0 else "spin_glass"


def game_phase(alpha: float, beta: float, eps: float) -> str:
    """Expected dynamical phase of the learning map (paper, Fig. 1 and Eq. 20; J0 = 0)."""
    if beta < INF and 1.0 / beta > Tc(eps):
        return "paramagnet"
    if alpha >= 1.0 and eps < 1.0:
        return "cycles"
    if eps > eps_c_estimate():
        return "chaotic"
    return "fixed_points"


@dataclass(frozen=True)
class Regime:
    name: str
    eps: float
    J0: float
    alpha: float
    beta: float
    note: str = ""

    @property
    def problem_key(self) -> str:
        return problem_key(self.eps, self.J0)

    @property
    def planner_phase(self) -> str:
        return planner_phase(self.eps, self.J0)

    @property
    def game_phase(self) -> str:
        return game_phase(self.alpha, self.beta, self.eps)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["beta"] = "inf" if self.beta == INF else self.beta
        d.update(problem_key=self.problem_key, planner_phase=self.planner_phase,
                 game_phase=self.game_phase, J0_eff=j0_eff(self.eps, self.J0))
        return d


def problem_key(eps: float, J0: float) -> str:
    return f"eps{eps:g}_J0{J0:g}"


REGIMES: tuple[Regime, ...] = (
    Regime("ferro_easy", eps=0.0, J0=3.0, alpha=0.1, beta=INF,
           note="cooperative couplings: easy for every optimiser and for the game"),
    Regime("sg_boundary", eps=0.0, J0=1.0, alpha=0.1, beta=INF,
           note="spin-glass / ferromagnet boundary: intermediate"),
    Regime("sg_satisficing", eps=0.0, J0=0.0, alpha=0.1, beta=INF,
           note="pure SK: hard optimum; the game freezes on sub-optimal fixed points"),
    Regime("sg_cycles", eps=0.0, J0=0.0, alpha=1.0, beta=INF,
           note="same instances as sg_satisficing, no memory: 2-cycles, never optimal"),
    Regime("nonreciprocal_chaos", eps=1.0, J0=0.0, alpha=0.1, beta=INF,
           note="optimum as hard as SK (W = J^S/2); the game is chaotic"),
    Regime("paramagnet", eps=0.0, J0=0.0, alpha=0.1, beta=0.25,
           note="same instances as sg_satisficing, T = 4 > T_c = 2: random play"),
)


def regimes_by_name(names=None) -> list[Regime]:
    if not names:
        return list(REGIMES)
    table = {r.name: r for r in REGIMES}
    unknown = [n for n in names if n not in table]
    if unknown:
        raise KeyError(f"unknown regimes {unknown}; available: {sorted(table)}")
    return [table[n] for n in names]


def problem_classes(regimes) -> list[tuple[str, float, float]]:
    """Distinct (key, eps, J0), in first-appearance order."""
    seen = {}
    for r in regimes:
        seen.setdefault(r.problem_key, (r.problem_key, r.eps, r.J0))
    return list(seen.values())
