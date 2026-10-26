"""Scaling benchmark: when does each optimiser stop finding the planner's optimum?

For every N in a grid and every instance J (eps fixed):
  * reference value R_ref: exact enumeration if N <= n_exact, otherwise the best value found
    by ANY solver on that instance (plus an optional extra-long reference run). Above n_exact
    the reference is only "best known": breakdown is then relative to the ensemble of solvers.
  * every solver is run n_runs times (independent seeds); we record best reward per run, time,
    work (spin-flip evaluations).
Derived metrics per (solver, N):
  * p_success  = fraction of runs reaching R_ref within tol
  * residual   = (R_ref - R_run) / |R_ref|   (median and mean)
  * TTS_99     = t_run * log(0.01) / log(1 - p_success)  (inf if p = 0, t_run if p = 1)
  * N_break    = smallest N where median p_success < p_threshold (the solver's breakdown size)
  * gap_to_parisi: R_ref(N) vs R_inf - A N^{-2/3}
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .problem import PlannerProblem
from .solvers import Solver, ExactEnumeration, ParallelTempering
from ..model import InteractionEnsemble


@dataclass
class BenchmarkConfig:
    Ns: tuple = (8, 12, 16, 20, 24, 32, 48, 64)
    eps: float = 0.0
    n_instances: int = 4
    n_runs: int = 4
    n_exact: int = 24
    tol: float = 1e-9
    p_threshold: float = 0.5
    seed: int = 7
    ref_sweeps: int = 3000          # extra-long parallel-tempering reference run (N > n_exact)
    ref_restarts: int = 2


@dataclass
class BenchmarkResult:
    solvers: list
    Ns: np.ndarray
    reward: np.ndarray        # (n_solvers, n_N, n_inst, n_runs)
    seconds: np.ndarray       # same
    evals: np.ndarray         # same
    ref: np.ndarray           # (n_N, n_inst)
    ref_exact: np.ndarray     # (n_N,) bool
    upper_bound: np.ndarray   # (n_N, n_inst) SDP bound (nan if not computed)
    extra: dict = field(default_factory=dict)

    def metrics(self, tol: float = 1e-9, p_threshold: float = 0.5) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._metrics(tol, p_threshold)

    def _metrics(self, tol, p_threshold):
        S, nN = len(self.solvers), len(self.Ns)
        ok = self.reward >= self.ref[None, :, :, None] - tol * np.maximum(1, np.abs(self.ref[None, :, :, None]))
        valid = np.isfinite(self.reward)
        p = np.where(valid.any(-1), np.nanmean(np.where(valid, ok, np.nan), axis=-1), np.nan)  # (S,N,inst)
        p_med = np.nanmedian(p, axis=-1)
        res = (self.ref[None, :, :, None] - self.reward) / np.abs(self.ref[None, :, :, None])
        res_med = np.nanmedian(res.reshape(S, nN, -1), axis=-1)
        t_run = np.nanmedian(self.seconds.reshape(S, nN, -1), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pm = np.clip(np.nanmean(p, axis=-1), 0, 1)
            tts = np.where(pm >= 1, t_run, np.where(pm <= 0, np.inf, t_run * np.log(0.01) / np.log1p(-pm)))
        n_break = []
        for s in range(S):
            bad = np.where((p_med[s] < p_threshold) & np.isfinite(p_med[s]))[0]
            n_break.append(int(self.Ns[bad[0]]) if len(bad) else -1)
        cert_gap = (self.upper_bound - self.ref) / np.abs(self.ref)
        return {"p_success": p, "p_median": p_med, "residual_median": res_med, "t_run": t_run,
                "tts99": tts, "N_break": np.array(n_break), "certification_gap": np.nanmedian(cert_gap, -1)}


def run_benchmark(solvers: list[Solver], cfg: BenchmarkConfig, log=print) -> BenchmarkResult:
    rng = np.random.default_rng(cfg.seed)
    nS, nN = len(solvers), len(cfg.Ns)
    shape = (nS, nN, cfg.n_instances, cfg.n_runs)
    R = np.full(shape, np.nan)
    T = np.full(shape, np.nan)
    E = np.full(shape, np.nan)
    ref = np.full((nN, cfg.n_instances), np.nan)
    ref_exact = np.zeros(nN, bool)
    ub = np.full((nN, cfg.n_instances), np.nan)
    exact = ExactEnumeration(max_N=cfg.n_exact)
    reference = ParallelTempering(sweeps=cfg.ref_sweeps, n_replicas=24, beta_min=0.1, beta_max=6.0)
    # warm-up: JIT-compile every kernel so that compile time does not pollute timings
    warm = PlannerProblem(InteractionEnsemble(8, cfg.eps).sample(np.random.default_rng(0), 1)[0])
    for sol in solvers + [exact]:
        if sol.applicable(8):
            sol.solve(warm, 0)
    ref_source = np.empty((nN, cfg.n_instances), dtype=object)
    for a, N in enumerate(cfg.Ns):
        t0 = time.time()
        for b in range(cfg.n_instances):
            P = PlannerProblem(InteractionEnsemble(N, cfg.eps).sample(rng, 1)[0], name=f"N{N}_i{b}")
            if N <= cfg.n_exact:
                ref[a, b] = exact.solve(P).reward
                ref_exact[a] = True
            for s, sol in enumerate(solvers):
                if sol.exact or not sol.applicable(N):
                    continue
                for r in range(cfg.n_runs):
                    out = sol.solve(P, rng.integers(2**63))
                    R[s, a, b, r], T[s, a, b, r], E[s, a, b, r] = out.reward, out.seconds, out.evals
                    if "upper_bound_reward" in out.info:
                        ub[a, b] = np.nanmin([ub[a, b], out.info["upper_bound_reward"]])
            if not ref_exact[a]:
                cand = [np.nanmax(R[:, a, b, :])]
                for k in range(cfg.ref_restarts):
                    cand.append(reference.solve(P, rng.integers(2**63)).reward)
                ref[a, b] = np.nanmax(cand)
                ref_source[a, b] = "reference_pt" if np.nanargmax(cand) > 0 else "best_solver"
            else:
                ref_source[a, b] = "exact"
                # sanity: no heuristic may beat the certified optimum
                assert np.nanmax(R[:, a, b, :]) <= ref[a, b] + 1e-9
        # exact solver row: fill with ref when exact
        for s, sol in enumerate(solvers):
            if sol.exact and sol.applicable(N):
                R[s, a] = ref[a][:, None]
        log(f"  benchmark N={N:4d}: {time.time() - t0:6.1f}s  ref(mean)={np.nanmean(ref[a]):.4f}")
    return BenchmarkResult([s.name for s in solvers], np.asarray(cfg.Ns), R, T, E, ref, ref_exact, ub,
                           extra={"ref_source": ref_source.astype(str)})
