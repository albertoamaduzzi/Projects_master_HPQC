"""SK-game vs quantum annealing: a staged, resumable campaign over N and over regimes.

For every regime (skgame.optim.regimes.REGIMES: easy / hard planner optimum x game phase) and
every N on a linear grid, on the same instances J:

  * reference optimum of the planner, max_S (1/N) S^T J S: exact (Gray code) for N <= n_exact,
    best known above (long parallel tempering + every solver, + the annealer at analysis time);
  * classical optimisers (greedy, SA, PT, tau-EO, SDP+GW, dimod SA / tabu);
  * the quantum annealer through dimod (D-Wave QPU via DWaveCliqueSampler, or the classical
    simulated-annealing sampler with the same interface for dry runs);
  * the game itself (learning dynamics at the regime's alpha, beta, eps): its phase and whether
    the state it settles in is the optimum.

Stages (each one resumable: finished files are skipped; run them in this order, the QPU stage can
run in parallel with `classical`):

  python main_simulation.py init       --preset full --out results/full
  python main_simulation.py plan       --out results/full       # sizes, units, QPU budget
  python main_simulation.py classical  --out results/full [--task i]   # compute nodes
  python main_simulation.py qpu        --out results/full       # needs internet + Leap token
  python main_simulation.py sim        --out results/full       # same, simulated annealing, no token
  python main_simulation.py analyze    --out results/full       # summary.csv + figures

`classical --task i` runs only the i-th problem class (Slurm array index). The QPU stage only
reads instances and calls the sampler, so it is cheap on CPU and can run on a login node.
See README_CINECA.md for the exact procedure on Leonardo.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

import numpy as np

from skgame.model import SKGame, GameParams, InteractionEnsemble, WindowRecorder
from skgame.observables import classify_phase
from skgame.optim.problem import PlannerProblem
from skgame.optim.regimes import REGIMES, regimes_by_name, problem_classes
from skgame.optim.solvers import ExactEnumeration, ParallelTempering, default_solvers

INF = float("inf")
TOL = 1e-9


# ============================================================================= config ===
@dataclass
class CampaignConfig:
    preset: str = "smoke"
    Ns: list = field(default_factory=lambda: [8, 12, 16, 20])
    regimes: list = field(default_factory=lambda: [r.name for r in REGIMES])
    n_instances: int = 2
    n_runs: int = 2                 # independent runs of every classical solver per instance
    n_exact: int = 20               # exact enumeration up to this N
    solver_budget: str = "quick"    # skgame.optim.solvers.default_solvers budget
    ref_sweeps: int = 300           # parallel-tempering reference above n_exact
    ref_restarts: int = 1
    game_runs: int = 4              # initial conditions of the game per instance
    t_alpha: int = 100              # game equilibration: t_alpha / alpha steps
    window: int = 32                # game observation window (phase + reward)
    num_reads: int = 100            # annealer reads per call
    annealing_time: float = 20.0    # microseconds (QPU only)
    chain_strength: float | None = None   # None: DWaveCliqueSampler default
    sim_num_reads: int = 100        # reads per call when the annealer is the classical simulator
    seed: int = 2024


PRESETS = {
    # a minute on a laptop: checks the whole pipeline
    "smoke": dict(),
    # tens of minutes on one core; N up to 64 with step 4
    "small": dict(Ns=list(range(8, 65, 4)), n_instances=4, n_runs=3, n_exact=24, ref_sweeps=2000,
                  ref_restarts=2, game_runs=8, t_alpha=300, window=64, num_reads=500),
    # the production campaign: N = 8, 16, ..., 160 (QPU clique limit ~170 on Advantage)
    "full": dict(Ns=list(range(8, 161, 8)), n_instances=5, n_runs=5, n_exact=24, solver_budget="paper",
                 ref_sweeps=5000, ref_restarts=2, game_runs=16, t_alpha=300, window=64, num_reads=1000),
}


def parse_Ns(text: str) -> list:
    """'8:160:8' -> 8, 16, ..., 160 (inclusive); '8,12,20' -> explicit list."""
    if ":" in text:
        a, b, s = (int(x) for x in text.split(":"))
        return list(range(a, b + 1, s))
    return [int(x) for x in text.split(",")]


def make_config(preset: str, **overrides) -> CampaignConfig:
    kw = dict(PRESETS[preset])
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return CampaignConfig(preset=preset, **kw)


def save_config(out: Path, cfg: CampaignConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "config.json", asdict(cfg))


def load_config(out: Path) -> CampaignConfig:
    path = out / "config.json"
    if not path.exists():
        sys.exit(f"{path} not found: run `python main_simulation.py init --preset ... --out {out}` first")
    d = json.loads(path.read_text())
    return CampaignConfig(**{f.name: d[f.name] for f in fields(CampaignConfig) if f.name in d})


# ============================================================================== files ===
def _write_json(path: Path, obj) -> None:
    """Atomic write: a killed job never leaves a half-written file that resume would trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def instance_path(out: Path, key: str, N: int) -> Path:
    return out / "instances" / key / f"N{N:04d}.npz"


def classical_path(out: Path, key: str, N: int) -> Path:
    return out / "classical" / key / f"N{N:04d}.json"


def game_path(out: Path, regime: str, N: int) -> Path:
    return out / "game" / regime / f"N{N:04d}.json"


def anneal_path(out: Path, sampler: str, key: str, N: int, b: int) -> Path:
    return out / "anneal" / sampler / key / f"N{N:04d}_i{b}.json"


# ========================================================================== instances ===
def _classes(cfg):
    return problem_classes(regimes_by_name(cfg.regimes))


def ensure_instances(out: Path, cfg: CampaignConfig, key: str, eps: float, J0: float, N: int) -> np.ndarray:
    """J for (problem class, N): (n_instances, N, N). Deterministic in (seed, class, N), saved once."""
    path = instance_path(out, key, N)
    if path.exists():
        return np.load(path)["J"]
    k = [c[0] for c in _classes(cfg)].index(key)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, k, N]))
    J = InteractionEnsemble(N, eps, J0=J0).sample(rng, cfg.n_instances)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, J=J, eps=eps, J0=J0, N=N)
    os.replace(tmp, path)
    return J


# ========================================================================== classical ===
def run_game(J: np.ndarray, regime, cfg: CampaignConfig, seed) -> dict:
    """Run the learning dynamics on every instance from cfg.game_runs initial conditions."""
    n_inst, N, _ = J.shape
    Jr = np.repeat(J, cfg.game_runs, axis=0)
    p = GameParams(N=N, alpha=regime.alpha, beta=regime.beta, eps=regime.eps, J0=regime.J0)
    t0 = time.perf_counter()
    g = SKGame(p, R=len(Jr), J=Jr, seed=seed)
    g.run(int(cfg.t_alpha / regime.alpha))
    rec = WindowRecorder(g.t, cfg.window, store_m=False)
    g.run(cfg.window, [rec])
    S = rec.result()["S"]                                        # (R, window, N) int8
    Sf = S.astype(float)
    reward = np.einsum("rti,rij,rtj->rt", Sf, Jr, Sf) / N       # R_N(t) along the window
    L = classify_phase(S)
    shape = (n_inst, cfg.game_runs)
    return {"best_reward": reward.max(1).reshape(shape).tolist(),
            "mean_reward": reward.mean(1).reshape(shape).tolist(),
            "final_reward": reward[:, -1].reshape(shape).tolist(),
            "period": L.reshape(shape).tolist(),                  # 1 fixed point, L cycle, 0 none
            "seconds": time.perf_counter() - t0, "steps": int(g.t)}


def stage_classical(out: Path, cfg: CampaignConfig, keys=None, log=print) -> None:
    classes = [c for c in _classes(cfg) if keys is None or c[0] in keys]
    regimes = regimes_by_name(cfg.regimes)
    solvers = [s for s in default_solvers(cfg.solver_budget) if not s.exact]
    exact = ExactEnumeration(max_N=max(cfg.n_exact, 8))
    reference = ParallelTempering(sweeps=cfg.ref_sweeps, n_replicas=24, beta_min=0.1, beta_max=6.0)
    warm = PlannerProblem(InteractionEnsemble(8).sample(np.random.default_rng(0), 1)[0])
    for s in solvers + [exact, reference]:                       # JIT-compile before timing
        s.solve(warm, 0)
    for ci, (key, eps, J0) in enumerate(classes):
        mine = [r for r in regimes if r.problem_key == key]
        for N in cfg.Ns:
            todo_game = [r for r in mine if not game_path(out, r.name, N).exists()]
            if classical_path(out, key, N).exists() and not todo_game:
                continue
            t0 = time.time()
            J = ensure_instances(out, cfg, key, eps, J0, N)
            if not classical_path(out, key, N).exists():
                rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1, ci, N]))
                rec = {"key": key, "eps": eps, "J0": J0, "N": N, "ref": [], "ref_exact": N <= cfg.n_exact,
                       "solvers": {s.name: {"reward": [], "seconds": []} for s in solvers},
                       "upper_bound": []}
                for b in range(cfg.n_instances):
                    P = PlannerProblem(J[b])
                    ub = np.nan
                    for s in solvers:
                        for _ in range(cfg.n_runs):
                            o = s.solve(P, rng.integers(2**63))
                            rec["solvers"][s.name]["reward"].append(o.reward)
                            rec["solvers"][s.name]["seconds"].append(o.seconds)
                            ub = np.fmin(ub, o.info.get("upper_bound_reward", np.nan))
                    best = max(v["reward"][-1 - r] for v in rec["solvers"].values() for r in range(cfg.n_runs))
                    if rec["ref_exact"]:
                        ref = exact.solve(P).reward
                        assert best <= ref + TOL * max(1, abs(ref)), "a heuristic beat the exact optimum"
                    else:
                        ref = max([best] + [reference.solve(P, rng.integers(2**63)).reward
                                            for _ in range(cfg.ref_restarts)])
                    rec["ref"].append(ref)
                    rec["upper_bound"].append(_nan_to_none(float(ub)))
                for v in rec["solvers"].values():          # (instance, run) layout
                    v["reward"] = np.reshape(v["reward"], (cfg.n_instances, cfg.n_runs)).tolist()
                    v["seconds"] = np.reshape(v["seconds"], (cfg.n_instances, cfg.n_runs)).tolist()
                _write_json(classical_path(out, key, N), rec)
            for r in todo_game:
                res = run_game(J, r, cfg, seed=np.random.SeedSequence([cfg.seed, 2, ci, N]))
                res.update(regime=r.as_dict(), N=N)
                _write_json(game_path(out, r.name, N), res)
            log(f"[classical] {key:12s} N={N:4d} done in {time.time() - t0:7.1f}s")


# ============================================================================ anneal ====
def make_annealer(cfg: CampaignConfig, sampler: str, solver_name=None):
    from skgame.optim.quantum import QuantumAnnealingSolver
    if sampler == "sim":
        return QuantumAnnealingSolver(num_reads=cfg.sim_num_reads, use_simulator=True, max_N=None)
    return QuantumAnnealingSolver(num_reads=cfg.num_reads, annealing_time=cfg.annealing_time, clique=True,
                                  chain_strength=cfg.chain_strength, solver_name=solver_name, max_N=None)


def anneal_units(out: Path, cfg: CampaignConfig, keys=None, max_N=None):
    """(key, eps, J0, N, b) still to do."""
    for key, eps, J0 in _classes(cfg):
        if keys is not None and key not in keys:
            continue
        for N in cfg.Ns:
            if max_N is not None and N > max_N:
                continue
            for b in range(cfg.n_instances):
                yield key, eps, J0, N, b


def stage_anneal(out: Path, cfg: CampaignConfig, sampler: str = "sim", keys=None, max_calls=None,
                 solver_name=None, check_only=False, log=print, retries=3) -> None:
    qa = make_annealer(cfg, sampler, solver_name)
    limit = qa.largest_problem()
    log(f"[anneal] sampler={sampler} solver={qa.solver_id()} largest_problem={limit}")
    units = list(anneal_units(out, cfg, keys))
    too_big = sorted({u[3] for u in units if limit is not None and u[3] > limit})
    if too_big:
        log(f"[anneal] skipping N={too_big}: above the largest embeddable clique ({limit})")
    todo = [u for u in units if (limit is None or u[3] <= limit)
            and not anneal_path(out, sampler, u[0], u[3], u[4]).exists()]
    log(f"[anneal] {len(units)} calls in the campaign, {len(todo)} still to do")
    if check_only:
        _estimate_qpu_time(qa, todo, cfg, log)
        return
    total_access = 0.0
    for n, (key, eps, J0, N, b) in enumerate(todo):
        if max_calls is not None and n >= max_calls:
            log(f"[anneal] stopped after --max-calls {max_calls}; rerun to continue")
            break
        J = ensure_instances(out, cfg, key, eps, J0, N)
        P = PlannerProblem(J[b])
        for attempt in range(retries + 1):
            try:
                res = qa.sample_problem(P, rng=np.random.SeedSequence([cfg.seed, 3, N, b]))
                break
            except Exception as exc:                     # network hiccups, solver busy, ...
                if attempt == retries:
                    raise
                log(f"[anneal] {key} N={N} i={b}: {type(exc).__name__}: {exc}; retry in {2 ** (attempt + 1)}s")
                time.sleep(2 ** (attempt + 1))
        res.update(key=key, eps=eps, J0=J0, N=N, instance=b, sampler=sampler)
        _write_json(anneal_path(out, sampler, key, N, b), res)
        acc = res["info"].get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
        total_access += acc
        log(f"[anneal] {n + 1:4d}/{len(todo)} {key:12s} N={N:4d} i={b} best={res['best_reward']:.5f} "
            f"wall={res['info']['wall_seconds']:.2f}s qpu_access={acc * 1e3:.1f}ms (total {total_access:.2f}s)")


def _estimate_qpu_time(qa, todo, cfg, log) -> None:
    """Estimate of the QPU access time still to spend, from the solver's own timing model."""
    solver = getattr(getattr(qa.sampler, "child", None), "solver", None)
    if solver is None or not hasattr(solver, "estimate_qpu_access_time"):
        log("[anneal] no timing model available (simulator?): nothing to estimate")
        return
    from minorminer.busclique import find_clique_embedding
    per_N = {}
    for N in sorted({u[3] for u in todo}):
        emb = find_clique_embedding(N, qa.sampler.target_graph, use_cache=True)
        n_qubits = sum(len(chain) for chain in emb.values())
        per_N[N] = solver.estimate_qpu_access_time(n_qubits, num_reads=cfg.num_reads,
                                                   annealing_time=cfg.annealing_time) / 1e6
        log(f"[anneal]   N={N:4d}: {n_qubits:5d} qubits, ~{per_N[N] * 1e3:.0f} ms per call")
    total = sum(per_N[u[3]] for u in todo)
    log(f"[anneal] estimated QPU access time for the {len(todo)} remaining calls: {total:.1f} s")


# ============================================================================ analyze ====
def _success(rewards, ref):
    return np.asarray(rewards, float) >= ref - TOL * max(1.0, abs(ref))


def collect(out: Path, cfg: CampaignConfig) -> list[dict]:
    """One row per (method, problem class or regime, N), against the best reference available."""
    rows = []
    samplers = sorted(p.name for p in (out / "anneal").glob("*")) if (out / "anneal").exists() else []
    regimes = regimes_by_name(cfg.regimes)
    for key, eps, J0 in _classes(cfg):
        for N in cfg.Ns:
            cp = classical_path(out, key, N)
            if not cp.exists():
                continue
            c = json.loads(cp.read_text())
            ref = np.array(c["ref"], float)
            ann = {s: [anneal_path(out, s, key, N, b) for b in range(cfg.n_instances)] for s in samplers}
            ann = {s: [json.loads(p.read_text()) if p.exists() else None for p in ps] for s, ps in ann.items()}
            ref_source = ["exact" if c["ref_exact"] else "classical"] * len(ref)
            for s, recs in ann.items():                       # the annealer may beat the reference
                for b, r in enumerate(recs):
                    if r is not None and r["best_reward"] > ref[b] + TOL * max(1, abs(ref[b])):
                        if c["ref_exact"]:
                            raise AssertionError(f"{s} beat the exact optimum: {key} N={N} i={b}")
                        ref[b], ref_source[b] = r["best_reward"], s
            base = {"problem": key, "eps": eps, "J0": J0, "N": N, "ref_exact": c["ref_exact"],
                    "ref_mean": float(ref.mean()), "ref_from_annealer": sum(x not in ("exact", "classical")
                                                                            for x in ref_source)}
            for name, v in c["solvers"].items():
                R = np.array(v["reward"], float)
                p = np.array([_success(R[b], ref[b]).mean() for b in range(len(ref))])
                res = (ref[:, None] - R) / np.abs(ref[:, None])
                rows.append(dict(base, kind="classical", method=name, regime="", p_success_median=np.median(p),
                                 p_success_mean=p.mean(), residual_median=float(np.median(res)),
                                 seconds_median=float(np.median(v["seconds"])), tts99=_tts(p.mean(), np.median(v["seconds"]))))
            for s, recs in ann.items():
                done = [(b, r) for b, r in enumerate(recs) if r is not None]
                if not done:
                    continue
                p = np.array([np.sum(np.array(r["counts"])[_success(r["rewards"], ref[b])]) / np.sum(r["counts"])
                              for b, r in done])
                best_res = np.array([(ref[b] - r["best_reward"]) / abs(ref[b]) for b, r in done])
                mean_res = np.array([np.average((ref[b] - np.array(r["rewards"])) / abs(ref[b]), weights=r["counts"])
                                     for b, r in done])
                t_read = np.median([_time_per_read(r) for b, r in done])
                acc = [r["info"].get("timing_us", {}).get("qpu_access_time", np.nan) / 1e6 for b, r in done]
                cbf = [r["info"].get("chain_break_fraction_mean", np.nan) for b, r in done]
                rows.append(dict(base, kind="anneal", method=f"anneal_{s}", regime="", n_done=len(done),
                                 p_success_median=float(np.median(p)), p_success_mean=float(p.mean()),
                                 p_best_found=float(np.mean(best_res <= TOL)),
                                 residual_median=float(np.median(best_res)), residual_mean_read=float(np.median(mean_res)),
                                 seconds_median=t_read, tts99=_tts(p.mean(), t_read),
                                 qpu_access_s=float(np.nansum(acc)), chain_break_fraction=_nan_to_none(float(np.mean(cbf)))))
            for r in regimes:
                if r.problem_key != key or not game_path(out, r.name, N).exists():
                    continue
                g = json.loads(game_path(out, r.name, N).read_text())
                best = np.array(g["best_reward"], float)
                p = np.array([_success(best[b], ref[b]).mean() for b in range(len(ref))])
                L = np.array(g["period"])
                rows.append(dict(base, kind="game", method="game", regime=r.name, planner_phase=r.planner_phase,
                                 game_phase=r.game_phase, p_success_median=float(np.median(p)),
                                 p_success_mean=float(p.mean()),
                                 residual_median=float(np.median((ref[:, None] - best) / np.abs(ref[:, None]))),
                                 residual_time_avg=float(np.median((ref[:, None] - np.array(g["mean_reward"]))
                                                                   / np.abs(ref[:, None]))),
                                 frac_fixed_point=float(np.mean(L == 1)), frac_cycle=float(np.mean(L > 1)),
                                 frac_no_recurrence=float(np.mean(L == 0)), seconds_median=g["seconds"]))
    return rows


def _time_per_read(r) -> float:
    """QPU: sampling time per read (anneal + readout + delay); simulator: wall time per read."""
    t = r["info"].get("timing_us", {})
    n = r["info"]["num_reads"]
    if "qpu_sampling_time" in t:
        return t["qpu_sampling_time"] / 1e6 / n
    return r["info"]["wall_seconds"] / n


def _tts(p: float, t: float) -> float:
    """Time to reach the reference with 99 % confidence: t ln(0.01) / ln(1 - p)."""
    if p >= 1:
        return float(t)
    if p <= 0:
        return INF
    return float(t * math.log(0.01) / math.log1p(-p))


def write_summary(out: Path, rows: list[dict]) -> Path:
    path = out / "summary" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def plot_summary(out: Path, rows: list[dict], show: bool = False) -> list[Path]:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "summary"
    figdir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(r["problem"] for r in rows))
    paths = []
    for metric, ylabel, log in [("p_success_median", "median success probability", False),
                                ("residual_median", "median residual (R_ref - R) / R_ref", True)]:
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False, sharey=True)
        for ax, key in zip(axes[0], keys):
            sub = [r for r in rows if r["problem"] == key]
            for label in dict.fromkeys((r["method"], r["regime"]) for r in sub):
                pts = sorted((r["N"], r[metric]) for r in sub if (r["method"], r["regime"]) == label)
                x, y = np.array(pts, float).T
                kind = next(r["kind"] for r in sub if (r["method"], r["regime"]) == label)
                style = {"anneal": dict(lw=2.5, marker="o"), "game": dict(ls="--", marker="s"),
                         "classical": dict(lw=1, alpha=0.6)}[kind]
                ax.plot(x, np.maximum(y, 1e-6) if log else y, label=label[1] or label[0], **style)
            ax.set_title(key)
            ax.set_xlabel("N")
            ax.legend(fontsize=6, loc="best")
            if log:
                ax.set_yscale("log")
        axes[0, 0].set_ylabel(ylabel)
        fig.tight_layout()
        paths.append(figdir / f"{metric}_vs_N.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig) if not show else plt.show()
    ann = [r for r in rows if r["kind"] == "anneal"]
    if ann:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for label in dict.fromkeys((r["method"], r["problem"]) for r in ann):
            pts = sorted((r["N"], r["tts99"]) for r in ann if (r["method"], r["problem"]) == label)
            x, y = np.array(pts, float).T
            ax.semilogy(x, y, marker="o", label=f"{label[0]} {label[1]}")
        ax.set_xlabel("N")
        ax.set_ylabel("TTS$_{99}$ per read time [s]")
        ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(figdir / "anneal_tts99_vs_N.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig) if not show else plt.show()
    game = [r for r in rows if r["kind"] == "game"]
    if game:
        regs = list(dict.fromkeys(r["regime"] for r in game))
        fig, axes = plt.subplots(1, len(regs), figsize=(3.4 * len(regs), 3.2), squeeze=False, sharey=True)
        for ax, reg in zip(axes[0], regs):
            pts = sorted((r["N"], r["frac_fixed_point"], r["frac_cycle"], r["frac_no_recurrence"])
                         for r in game if r["regime"] == reg)
            x, fp, cy, no = np.array(pts, float).T
            ax.stackplot(x, fp, cy, no, labels=["fixed point", "cycle", "no recurrence"], alpha=0.8)
            ax.set_title(reg, fontsize=9)
            ax.set_xlabel("N")
        axes[0, 0].set_ylabel("fraction of runs")
        axes[0, -1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        paths.append(figdir / "game_phases_vs_N.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig) if not show else plt.show()
    return paths


def stage_analyze(out: Path, cfg: CampaignConfig, log=print, show=False) -> list[dict]:
    rows = collect(out, cfg)
    if not rows:
        log("[analyze] nothing to analyse yet: run the classical stage first")
        return rows
    path = write_summary(out, rows)
    figs = plot_summary(out, rows, show=show)
    log(f"[analyze] {len(rows)} rows -> {path}")
    for f in figs:
        log(f"[analyze] figure {f}")
    return rows


# =============================================================================== plan ====
def describe_plan(out: Path, cfg: CampaignConfig, log=print) -> None:
    classes = _classes(cfg)
    log(f"campaign '{cfg.preset}' in {out}")
    log(f"  N grid ({len(cfg.Ns)} values): {cfg.Ns}")
    log(f"  instances per (class, N): {cfg.n_instances}; exact reference for N <= {cfg.n_exact}")
    log("  regimes:")
    for r in regimes_by_name(cfg.regimes):
        d = r.as_dict()
        log(f"    {r.name:20s} eps={r.eps:<4g} J0={r.J0:<4g} alpha={r.alpha:<4g} beta={d['beta']!s:<5} "
            f"planner={r.planner_phase:13s} game={r.game_phase:12s} | {r.note}")
    log("  classical tasks (Slurm array index -> problem class):")
    for i, (key, *_rest) in enumerate(classes):
        log(f"    --task {i}: {key}")
    n_call = len(cfg.Ns) * len(classes) * cfg.n_instances
    # t_call ~ t_programming + num_reads (t_anneal + t_readout + t_delay), Advantage typical values
    t_prog, t_ro, t_d = 15e-3, 100e-6, 20e-6
    t_read = cfg.annealing_time * 1e-6 + t_ro + t_d
    t_call = t_prog + cfg.num_reads * t_read
    log(f"  annealer calls: |Ns| * classes * instances = {len(cfg.Ns)} * {len(classes)} * {cfg.n_instances} = {n_call}")
    log(f"  QPU time per call ~ {t_prog * 1e3:.0f} ms + {cfg.num_reads} * ({cfg.annealing_time:g} + "
        f"{t_ro * 1e6:.0f} + {t_d * 1e6:.0f}) us = {t_call * 1e3:.0f} ms")
    log(f"  QPU access time ~ {n_call} * {t_call:.3f} s = {n_call * t_call:.0f} s  "
        f"(upper bound; `qpu --check` gives the solver's own estimate)")


# ================================================================================ cli ====
def _task_keys(cfg, task, problem):
    classes = _classes(cfg)
    if task is not None:
        if not 0 <= task < len(classes):
            sys.exit(f"--task must be in 0..{len(classes) - 1}")
        return [classes[task][0]]
    if problem:
        return problem
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=Path(os.environ.get("SKGAME_OUT", "results/smoke")),
                        help="campaign directory (default: $SKGAME_OUT or results/smoke)")
    p = sub.add_parser("init", parents=[common], help="create the campaign directory and its config.json")
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--Ns", type=parse_Ns, help="override the N grid: '8:160:8' or '8,16,32'")
    p.add_argument("--regimes", nargs="+", help="subset of regime names")
    p.add_argument("--n-instances", type=int)
    p.add_argument("--num-reads", type=int)
    p.add_argument("--annealing-time", type=float)
    p.add_argument("--chain-strength", type=float)
    p.add_argument("--force", action="store_true", help="overwrite an existing config.json")
    sub.add_parser("plan", parents=[common], help="print sizes, tasks and the QPU budget")
    p = sub.add_parser("classical", parents=[common], help="references, classical solvers and the game")
    p.add_argument("--task", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID") and
                   int(os.environ["SLURM_ARRAY_TASK_ID"]), help="problem-class index (default: Slurm array id)")
    p.add_argument("--problem", nargs="+", help="problem-class keys (see `plan`)")
    for name, hlp in [("qpu", "D-Wave QPU (needs a Leap token and outbound HTTPS)"),
                      ("sim", "classical simulated-annealing stand-in (no token)")]:
        p = sub.add_parser(name, parents=[common], help=f"annealer stage: {hlp}")
        p.add_argument("--problem", nargs="+")
        p.add_argument("--max-calls", type=int, help="stop after this many sampler calls")
        if name == "qpu":
            p.add_argument("--solver", help="QPU name, e.g. Advantage_system4.1 (default: Leap's choice)")
            p.add_argument("--check", action="store_true",
                           help="connect, print solver and clique size, estimate QPU time; spends nothing")
    p = sub.add_parser("analyze", parents=[common], help="summary.csv and figures")
    a = ap.parse_args(argv)

    out = a.out
    if a.cmd == "init":
        if (out / "config.json").exists() and not a.force:
            cfg = load_config(out)
            print(f"{out}/config.json exists (preset '{cfg.preset}'); use --force to replace it")
        else:
            cfg = make_config(a.preset, Ns=a.Ns, regimes=a.regimes, n_instances=a.n_instances,
                              num_reads=a.num_reads, annealing_time=a.annealing_time, chain_strength=a.chain_strength)
            regimes_by_name(cfg.regimes)
            save_config(out, cfg)
        describe_plan(out, cfg)
        return
    cfg = load_config(out)
    if a.cmd == "plan":
        describe_plan(out, cfg)
    elif a.cmd == "classical":
        stage_classical(out, cfg, _task_keys(cfg, a.task, a.problem))
    elif a.cmd in ("qpu", "sim"):
        stage_anneal(out, cfg, a.cmd, keys=a.problem, max_calls=a.max_calls,
                     solver_name=getattr(a, "solver", None), check_only=getattr(a, "check", False))
    elif a.cmd == "analyze":
        stage_analyze(out, cfg)


if __name__ == "__main__":
    main()
