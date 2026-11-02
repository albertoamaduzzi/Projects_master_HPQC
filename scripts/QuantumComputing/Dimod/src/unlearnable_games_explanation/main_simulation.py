"""SK-game vs quantum annealing: a staged, resumable campaign over N and over regimes.

For every regime (skgame.optim.regimes.REGIMES: easy / hard planner optimum x game phase) and
every N on a linear grid, on the same instances J:

  * reference optimum of the planner, max_S (1/N) S^T J S: exact (Gray code) for N <= n_exact,
    best known above (long parallel tempering + every solver, + the annealer at analysis time);
  * classical optimisers (greedy, SA, PT, tau-EO, SDP+GW, dimod SA / tabu);
  * the quantum annealer through dimod, one or more back-ends (skgame.optim.backends):
    `qpu:zephyr:clique` (Advantage2) and `qpu:pegasus:clique` (Advantage) by default, same
    instances and settings, so `analyze` says which topology finds the optimum more often;
    `sim:...` back-ends run the same pipeline with classical simulated annealing (no token);
  * the game itself (learning dynamics at the regime's alpha, beta, eps): its phase and whether
    the state it settles in is the optimum.

Stages (each one resumable: finished files are skipped; run them in this order, the QPU stage can
run in parallel with `classical`):

  python main_simulation.py init       --preset full --out results/full
  python main_simulation.py plan       --out results/full       # sizes, units, QPU budget
  python main_simulation.py instances  --out results/full       # optional: write all J now
  python main_simulation.py classical  --out results/full [--task i]   # compute nodes
  python main_simulation.py embed      --out results/full       # embeddings + chain stats (no QPU time)
  python main_simulation.py qpu        --out results/full       # every qpu back-end; internet + Leap token
  python main_simulation.py sim        --out results/full       # sim:logical (or --backend sim:zephyr:clique)
  python main_simulation.py analyze    --out results/full       # summary.csv, figures, Zephyr vs Pegasus

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
    backends: list = field(default_factory=lambda: ["qpu:zephyr:clique", "qpu:pegasus:clique"])
    chain_strength_prefactor: float = 1.414   # chain strength = prefactor x uniform torque compensation
    chain_strength_by_N: dict = field(default_factory=dict)   # {backend label: {N: p}} from coupling-scan
    chain_strength_by_class: dict = field(default_factory=dict)  # {problem class: {backend label: {N: p}}}
    extended_j_range: bool = False  # chain couplers in the QPU's extended range (less compression)
    flux_biases_files: dict = field(default_factory=dict)     # {backend label: shims/.../fbos.json} from shim
    chain_break_method: str = "majority_vote"  # majority_vote | minimize_energy | weighted_random | discard
    polish: bool = True             # also score every read after greedy single-flip descent
    sim_num_reads: int = 100        # reads per call when the annealer is the classical simulator
    sim_num_sweeps: int = 1000
    seed: int = 2024


PRESETS = {
    # a minute on a laptop: checks the whole pipeline
    "smoke": dict(),
    # tens of minutes on one core; N up to 64 with step 4
    "small": dict(Ns=list(range(8, 65, 4)), n_instances=4, n_runs=3, n_exact=24, ref_sweeps=2000,
                  ref_restarts=2, game_runs=8, t_alpha=300, window=64, num_reads=500),
    # the production campaign: N = 8, 16, ..., 160 (clique limit ~180 on ideal Zephyr/Pegasus,
    # a bit less on real chips: `qpu --check` prints it and larger N are skipped)
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


def _spins(S) -> str:
    """Configuration as a string of + and - (agent 0 first)."""
    return "".join("+" if x > 0 else "-" for x in np.asarray(S).ravel())


def _canonical(spins: str) -> str:
    """S and -S have the same energy (no field): fix the sign so that agent 0 plays +."""
    return spins if not spins or spins[0] == "+" else spins.translate(str.maketrans("+-", "-+"))


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
                rec = {"key": key, "eps": eps, "J0": J0, "N": N, "ref": [], "ref_S": [], "ref_source": [],
                       "ref_exact": N <= cfg.n_exact,
                       "solvers": {s.name: {"reward": [], "seconds": []} for s in solvers},
                       "upper_bound": []}
                for b in range(cfg.n_instances):
                    P = PlannerProblem(J[b])
                    ub = np.nan
                    best, best_S, best_src = -INF, None, None
                    for s in solvers:
                        for _ in range(cfg.n_runs):
                            o = s.solve(P, rng.integers(2**63))
                            rec["solvers"][s.name]["reward"].append(o.reward)
                            rec["solvers"][s.name]["seconds"].append(o.seconds)
                            ub = np.fmin(ub, o.info.get("upper_bound_reward", np.nan))
                            if o.reward > best:
                                best, best_S, best_src = o.reward, o.S, s.name
                    if rec["ref_exact"]:
                        ex = exact.solve(P)
                        assert best <= ex.reward + TOL * max(1, abs(ex.reward)), "a heuristic beat the exact optimum"
                        ref, ref_S, src = ex.reward, ex.S, "exact"
                    else:
                        ref, ref_S, src = best, best_S, best_src
                        for _ in range(cfg.ref_restarts):
                            o = reference.solve(P, rng.integers(2**63))
                            if o.reward > ref:
                                ref, ref_S, src = o.reward, o.S, "reference_pt"
                    rec["ref"].append(ref)
                    rec["ref_S"].append(_spins(ref_S))
                    rec["ref_source"].append(src)
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
def anneal_settings(cfg: CampaignConfig, solver_name=None):
    from skgame.optim.backends import AnnealSettings
    return AnnealSettings(num_reads=cfg.num_reads, annealing_time=cfg.annealing_time,
                          chain_strength_prefactor=cfg.chain_strength_prefactor,
                          chain_break_method=cfg.chain_break_method, polish=cfg.polish,
                          sim_num_reads=cfg.sim_num_reads, sim_num_sweeps=cfg.sim_num_sweeps,
                          solver_name=solver_name, extended_j_range=cfg.extended_j_range)


def make_backend(out: Path, cfg: CampaignConfig, spec: str, solver_name=None):
    from skgame.optim.backends import AnnealBackend, label
    be = AnnealBackend(spec, anneal_settings(cfg, solver_name), cache_root=out / "embeddings")
    f = (cfg.flux_biases_files or {}).get(label(spec))
    if f and (out / f).exists():                         # flux-bias offsets learned by `shim`
        d = json.loads((out / f).read_text())
        be.flux_biases = {int(k): float(v) for k, v in d["fbos"].items()}
        be.flux_biases_source = f
    return be


def anneal_units(out: Path, cfg: CampaignConfig, keys=None, max_N=None, Ns=None):
    """(key, eps, J0, N, b) to do, N by N (all classes and instances of one N before the next), so that
    a run stopped by the budget leaves complete N behind. `Ns` restricts to some N."""
    for N in cfg.Ns:
        if (max_N is not None and N > max_N) or (Ns is not None and N not in Ns):
            continue
        for key, eps, J0 in _classes(cfg):
            if keys is not None and key not in keys:
                continue
            for b in range(cfg.n_instances):
                yield key, eps, J0, N, b


def stage_embed(out: Path, cfg: CampaignConfig, specs, solver_name=None, log=print) -> list[dict]:
    """Compute (and cache) the embedding of every N for every back-end; no QPU time is used.
    For qpu back-ends this needs the token (to read the real working graph)."""
    rows = []
    for spec in specs:
        be = make_backend(out, cfg, spec, solver_name)
        if not be.embedded:
            continue
        limit = be.largest_problem()
        log(f"[embed] {spec}: graph {be.graph_id()} largest K_N = {limit}")
        for N in cfg.Ns:
            if limit is not None and N > limit:
                log(f"[embed]   N={N:4d}: too large")
                continue
            from skgame.optim.embedding import chain_stats
            st = chain_stats(be.embedding(N))
            rows.append(dict(backend=spec, graph=be.graph_id(), N=N, largest_N=limit, **st))
            log(f"[embed]   N={N:4d}: {st['n_qubits']:5d} qubits, chain length mean {st['chain_len_mean']:.2f}"
                f" max {st['chain_len_max']}")
    if rows:
        path = out / "summary" / "embeddings.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"[embed] -> {path}")
    return rows


class QPUBudget:
    """Hard cap on the QPU access time spent by one command (all back-ends together).

    Before each call the cost is predicted with the solver's own timing model
    (`estimate_qpu_access_time`) and inflated by a safety factor; the call is made only if
    spent + inflated estimate <= limit. The factor starts at `safety` and grows if real
    charges come out larger than predicted (factor >= 1.1 x worst observed actual/estimate).
    After each call the REAL `qpu_access_time` returned by the QPU is added to `spent`.
    """

    def __init__(self, limit_s: float | None, safety: float = 1.5):
        self.limit, self.safety, self.spent, self.worst_ratio = limit_s, safety, 0.0, 0.0

    def factor(self) -> float:
        return max(self.safety, 1.1 * self.worst_ratio)

    def allows(self, estimate_s: float) -> bool:
        return self.limit is None or self.spent + self.factor() * estimate_s <= self.limit

    def charge(self, actual_s: float, estimate_s: float) -> None:
        self.spent += actual_s
        if estimate_s > 0 and actual_s > 0:
            self.worst_ratio = max(self.worst_ratio, actual_s / estimate_s)


def _call_estimate(be, N: int, cfg: CampaignConfig) -> float:
    """Predicted QPU access time of one call [s]. Without a timing model (mock QPU) a
    deliberately pessimistic formula: 20 ms programming + reads x (anneal + 300 us)."""
    est = be.estimate_access_time(N) if be.device == "qpu" else 0.0
    if est is None:
        est = 0.020 + cfg.num_reads * (cfg.annealing_time * 1e-6 + 300e-6)
    return float(est)


def _transient(exc: Exception) -> bool:
    """Worth retrying? Network / service problems yes; errors in the problem or the embedding
    (they would fail identically every time, and fail before any QPU time is used) no."""
    try:
        from minorminer.utils.exceptions import EmbeddingError
        if isinstance(exc, EmbeddingError):
            return False
    except ImportError:  # pragma: no cover
        pass
    return not isinstance(exc, (ValueError, KeyError, TypeError, AssertionError))


def stage_anneal(out: Path, cfg: CampaignConfig, spec: str = "sim:logical", keys=None, max_calls=None,
                 solver_name=None, check_only=False, log=print, retries=3, backend=None,
                 budget: QPUBudget | None = None, Ns=None) -> None:
    be = backend or make_backend(out, cfg, spec, solver_name)
    limit = be.largest_problem()
    log(f"[anneal] backend={be.spec} solver={be.solver_id()} largest_problem={limit}")
    units = list(anneal_units(out, cfg, keys, Ns=Ns))
    too_big = sorted({u[3] for u in units if limit is not None and u[3] > limit})
    if too_big:
        log(f"[anneal] skipping N={too_big}: above the largest embeddable clique ({limit})")
    todo = [u for u in units if (limit is None or u[3] <= limit)
            and not anneal_path(out, be.label, u[0], u[3], u[4]).exists()]
    log(f"[anneal] {len(units)} calls in the campaign, {len(todo)} still to do")
    if check_only:
        _estimate_qpu_time(be, todo, log)
        return
    total_access = 0.0
    budget = budget or QPUBudget(None)
    for n, (key, eps, J0, N, b) in enumerate(todo):
        if max_calls is not None and n >= max_calls:
            log(f"[anneal] stopped after --max-calls {max_calls}; rerun to continue")
            break
        est = _call_estimate(be, N, cfg)
        if not budget.allows(est):
            log(f"[anneal] budget: spent {budget.spent:.3f}s of {budget.limit:.3f}s; next call (N={N}) estimated "
                f"{est:.3f}s x safety {budget.factor():.2f} would exceed it -> stopped (rerun to continue)")
            break
        J = ensure_instances(out, cfg, key, eps, J0, N)
        P = PlannerProblem(J[b])
        if cfg.chain_strength_by_N or cfg.chain_strength_by_class:   # p chosen by calibration / coupling-scan
            from coupling_scan import prefactor_for
            be.settings.chain_strength_prefactor = prefactor_for(cfg, be.label, key, N)
        for attempt in range(retries + 1):
            try:
                res = be.sample(P, rng=np.random.SeedSequence([cfg.seed, 3, N, b]),
                                label=f"skgame {out.name} {be.label} {key} N={N} i={b}")
                break
            except Exception as exc:                     # network hiccups, solver busy, ...
                if attempt == retries or not _transient(exc):
                    raise
                log(f"[anneal] {key} N={N} i={b}: {type(exc).__name__}: {exc}; retry in {2 ** (attempt + 1)}s")
                time.sleep(2 ** (attempt + 1))
        res.update(key=key, eps=eps, J0=J0, N=N, instance=b, backend=be.spec)
        res["info"]["estimated_access_s"] = est
        phys = res.pop("_physical", None)
        if phys is not None:                     # the exact physical problem sent (before auto_scale)
            pp = anneal_path(out, be.label, key, N, b).with_name(f"N{N:04d}_i{b}_physical.npz")
            pp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pp, **phys)
            res["info"]["physical_problem_file"] = pp.name
        _write_json(anneal_path(out, be.label, key, N, b), res)
        acc = res["info"].get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
        budget.charge(acc if acc > 0 else est, est)
        total_access += acc
        cbf = res["info"].get("chain_break_fraction_mean")
        log(f"[anneal] {n + 1:4d}/{len(todo)} {be.label} {key:10s} N={N:4d} i={b} best={res['best_reward']:.5f} "
            f"chain_breaks={'-' if cbf is None else f'{cbf:.3f}'} qpu_access={acc * 1e3:.1f}ms (est {est * 1e3:.0f}ms) "
            f"(total {total_access:.3f}s)")


def qpu_spent(out: Path, log=print) -> dict:
    """QPU access time already spent in a campaign, per back-end, from the saved results
    (the value D-Wave charged for each call), with the ratio to the prediction."""
    rows = {}
    for f in sorted((out / "anneal").glob("qpu_*/*/*.json")) if (out / "anneal").exists() else []:
        info = json.loads(f.read_text())["info"]
        r = rows.setdefault(f.parent.parent.name, {"calls": 0, "spent_s": 0.0, "est_s": 0.0})
        r["calls"] += 1
        r["spent_s"] += info.get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
        r["est_s"] += info.get("estimated_access_s", 0.0)
    total = sum(r["spent_s"] for r in rows.values())
    for name, r in rows.items():
        ratio = f"{r['spent_s'] / r['est_s']:.2f}" if r["est_s"] else "n/a"
        log(f"[spent] {name:22s} {r['calls']:4d} calls  {r['spent_s']:8.3f} s  (actual/estimate = {ratio})")
    log(f"[spent] total QPU access time in {out}: {total:.3f} s")
    return rows


def _estimate_qpu_time(be, todo, log) -> None:
    """QPU access time still to spend, from the solver's own timing model (no QPU time used)."""
    if be.device != "qpu":
        log("[anneal] not a QPU back-end: nothing to estimate")
        return
    per_N = {}
    for N in sorted({u[3] for u in todo}):
        per_N[N] = be.estimate_access_time(N)
        if per_N[N] is None:
            log("[anneal] the solver exposes no timing model")
            return
        log(f"[anneal]   N={N:4d}: ~{per_N[N] * 1e3:.0f} ms per call")
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
                    best = max(r["best_reward"], r.get("polished_best_reward", -INF)) if r is not None else -INF
                    if best > ref[b] + TOL * max(1, abs(ref[b])):
                        if c["ref_exact"]:
                            raise AssertionError(f"{s} beat the exact optimum: {key} N={N} i={b}")
                        ref[b], ref_source[b] = best, s
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
                p = np.array([np.sum(np.array(r["counts"])[_success(r["rewards"], ref[b])]) / max(1, np.sum(r["counts"]))
                              for b, r in done])
                best_res = np.array([(ref[b] - r["best_reward"]) / abs(ref[b]) for b, r in done])
                mean_res = np.array([np.average((ref[b] - np.array(r["rewards"])) / abs(ref[b]), weights=r["counts"])
                                     if np.sum(r["counts"]) else np.nan for b, r in done])
                t_read = np.median([_time_per_read(r) for b, r in done])
                acc = [r["info"].get("timing_us", {}).get("qpu_access_time", np.nan) / 1e6 for b, r in done]
                cbf = [r["info"].get("chain_break_fraction_mean", np.nan) for b, r in done]
                extra = {}
                if all("polished_rewards" in r for b, r in done):
                    pp = np.array([np.sum(np.array(r["polished_counts"])[_success(r["polished_rewards"], ref[b])])
                                   / max(1, np.sum(r["polished_counts"])) for b, r in done])
                    extra.update(p_success_polished=float(np.median(pp)),
                                 residual_polished=float(np.median([(ref[b] - r["polished_best_reward"]) / abs(ref[b])
                                                                    for b, r in done])))
                emb = [r["info"].get("embedding") for b, r in done]
                if all(emb):
                    extra.update(chain_len_mean=float(np.mean([e["chain_len_mean"] for e in emb])),
                                 chain_len_max=int(max(e["chain_len_max"] for e in emb)),
                                 n_qubits=int(np.mean([e["n_qubits"] for e in emb])),
                                 chain_strength=_nan_to_none(float(np.mean([r["info"].get("chain_strength") or np.nan
                                                                            for b, r in done]))))
                rows.append(dict(base, kind="anneal", method=f"anneal_{s}", regime="", n_done=len(done), **extra,
                                 p_success_median=float(np.median(p)), p_success_mean=float(p.mean()),
                                 p_best_found=float(np.mean(best_res <= TOL)),
                                 residual_median=float(np.median(best_res)), residual_mean_read=float(np.nanmedian(mean_res)) if np.isfinite(mean_res).any() else None,
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


def compare_backends(rows: list[dict]) -> list[dict]:
    """Per (problem class, N) where at least two annealing back-ends ran: who is better.
    Winner = higher median success probability per read; ties -> lower median residual."""
    ann = [r for r in rows if r["kind"] == "anneal"]
    out = []
    for device in ("qpu", "sim"):                      # rank QPU against QPU, simulator against simulator
        for key in dict.fromkeys(r["problem"] for r in ann):
            for N in sorted({r["N"] for r in ann if r["problem"] == key}):
                pts = [r for r in ann if r["problem"] == key and r["N"] == N
                       and r["method"].startswith(f"anneal_{device}_")]
                if len(pts) < 2:
                    continue
                best = max(pts, key=lambda r: (r["p_success_median"], -r["residual_median"]))
                tie = sum((r["p_success_median"], r["residual_median"]) == (best["p_success_median"], best["residual_median"])
                          for r in pts) > 1
                row = {"device": device, "problem": key, "N": N,
                       "winner": "tie" if tie else best["method"].removeprefix("anneal_")}
                for r in pts:
                    m = r["method"].removeprefix("anneal_")
                    for k in ("p_success_median", "p_success_polished", "residual_median", "tts99",
                              "chain_break_fraction", "chain_len_mean", "chain_strength"):
                        if k in r:
                            row[f"{m}:{k}"] = r[k]
                out.append(row)
    return out


def plot_backend_comparison(out: Path, comp: list[dict], rows: list[dict], show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ann = [r for r in rows if r["kind"] == "anneal"]
    methods = list(dict.fromkeys(r["method"] for r in ann))
    keys = list(dict.fromkeys(r["problem"] for r in ann))
    panels = [("p_success_median", "success probability per read"), ("residual_median", "residual of best read"),
              ("chain_break_fraction", "chain-break fraction"), ("chain_len_mean", "mean chain length")]
    fig, axes = plt.subplots(len(panels), len(keys), figsize=(3.8 * len(keys), 2.8 * len(panels)), squeeze=False)
    for j, key in enumerate(keys):
        for i, (metric, lab) in enumerate(panels):
            ax = axes[i, j]
            for m in methods:
                pts = sorted((r["N"], r.get(metric)) for r in ann if r["method"] == m and r["problem"] == key
                             and r.get(metric) is not None)
                if pts:
                    x, y = np.array(pts, float).T
                    ax.plot(x, y, "o-", ms=3, label=m.removeprefix("anneal_"))
            if i == 0:
                ax.set_title(key)
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(panels) - 1:
                ax.set_xlabel("N")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    path = out / "summary" / "backend_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig) if not show else plt.show()
    return path


def chain_break_table(out: Path, cfg: CampaignConfig) -> list[dict]:
    """Solution quality as a function of the number of broken chains in the read.

    One row per (back-end, problem class, N, number of broken chains): how many reads, and among
    them the success probability and the mean residual, raw and after greedy polishing. The
    reference is the same as in `collect` (exact / classical, improved by any annealer read);
    if the classical stage has not run yet it is the best read of all annealers ("best seen").
    """
    if not (out / "anneal").exists():
        return []
    labels = sorted(p.name for p in (out / "anneal").glob("*"))
    rows = []
    for key, eps, J0 in _classes(cfg):
        for N in cfg.Ns:
            recs = {lab: [json.loads(p.read_text()) if p.exists() else None
                          for p in (anneal_path(out, lab, key, N, b) for b in range(cfg.n_instances))]
                    for lab in labels}
            if not any(r for v in recs.values() for r in v):
                continue
            cp = classical_path(out, key, N)
            if cp.exists():
                c = json.loads(cp.read_text())
                ref, src = np.array(c["ref"], float), ("exact" if c["ref_exact"] else "classical")
            else:
                ref, src = np.full(cfg.n_instances, -INF), "best_seen"
            for v in recs.values():
                for b, r in enumerate(v):
                    if r is not None:
                        ref[b] = max(ref[b], r["best_reward"], r.get("polished_best_reward", -INF))
            for lab, v in recs.items():
                acc = {}
                for b, r in enumerate(v):
                    if r is None or "reads" not in r:
                        continue
                    t = r["reads"]
                    rw, nb, ct = (np.array(t[k], float) for k in ("reward", "n_broken", "count"))
                    pr = np.array(t["polished_reward"], float) if t.get("polished_reward") else None
                    ok = _success(rw, ref[b])
                    okp = _success(pr, ref[b]) if pr is not None else None
                    for k in np.unique(nb).astype(int):
                        m = nb == k
                        a = acc.setdefault(k, {"reads": 0, "succ": 0.0, "succ_p": 0.0, "res": 0.0, "res_p": 0.0})
                        a["reads"] += ct[m].sum()
                        a["succ"] += ct[m][ok[m]].sum()
                        a["res"] += np.sum(ct[m] * (ref[b] - rw[m]) / abs(ref[b]))
                        if pr is not None:
                            a["succ_p"] += ct[m][okp[m]].sum()
                            a["res_p"] += np.sum(ct[m] * (ref[b] - pr[m]) / abs(ref[b]))
                tot = sum(a["reads"] for a in acc.values())
                for k, a in sorted(acc.items()):
                    rows.append({"backend": lab, "problem": key, "N": N, "n_broken": k, "reads": int(a["reads"]),
                                 "frac_reads": a["reads"] / tot, "p_success": a["succ"] / a["reads"],
                                 "p_success_polished": a["succ_p"] / a["reads"],
                                 "residual_mean": a["res"] / a["reads"], "residual_polished_mean": a["res_p"] / a["reads"],
                                 "reference": src})
    return rows


def plot_chain_breaks(out: Path, rows: list[dict], min_reads: int = 5, show=False) -> Path | None:
    if not rows:
        return None
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(dict.fromkeys(r["backend"] for r in rows))
    panels = [("frac_reads", "fraction of reads"), ("residual_mean", "mean residual (raw read)"),
              ("residual_polished_mean", "mean residual after polish"), ("p_success", "P(read = reference)")]
    fig, axes = plt.subplots(len(panels), len(labs), figsize=(4.6 * len(labs), 2.7 * len(panels)), squeeze=False)
    cm = plt.get_cmap("viridis")
    Ns = sorted({r["N"] for r in rows})
    for j, lab in enumerate(labs):
        for i, (m, ylab) in enumerate(panels):
            ax = axes[i, j]
            for N in Ns:
                agg = {}
                for r in rows:                       # pool the problem classes
                    if r["backend"] == lab and r["N"] == N:
                        a = agg.setdefault(r["n_broken"], [0, 0.0])
                        a[0] += r["reads"]
                        a[1] += r[m] * r["reads"] if m != "frac_reads" else r["reads"]
                tot = sum(a[0] for a in agg.values()) or 1
                pts = sorted((k, (a[1] / tot) if m == "frac_reads" else a[1] / a[0]) for k, a in agg.items()
                             if m == "frac_reads" or a[0] >= min_reads)
                if pts:
                    x, y = np.array(pts, float).T
                    ax.plot(x, y, "o-", ms=3, color=cm(Ns.index(N) / max(1, len(Ns) - 1)), label=f"N={N}")
            if i == 0:
                ax.set_title(lab)
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(panels) - 1:
                ax.set_xlabel("number of broken chains in the read")
    axes[0, -1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = out / "summary" / "chain_breaks.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig) if not show else plt.show()
    return path


def report_chain_breaks(out: Path, cfg: CampaignConfig, log=print) -> list[dict]:
    rows = chain_break_table(out, cfg)
    if not rows:
        log("[chains] no per-read chain-break data (results produced before this feature, or no embedded back-end)")
        return rows
    path = out / "summary" / "chain_breaks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig = plot_chain_breaks(out, rows)
    for lab in dict.fromkeys(r["backend"] for r in rows):
        for N in sorted({r["N"] for r in rows if r["backend"] == lab}):
            sel = [r for r in rows if r["backend"] == lab and r["N"] == N]
            reads = sum(r["reads"] for r in sel)
            mean_nb = sum(r["n_broken"] * r["reads"] for r in sel) / reads
            z = sum(r["reads"] for r in sel if r["n_broken"] == 0) / reads
            log(f"[chains] {lab:22s} N={N:4d}: {reads:6d} reads, mean broken chains {mean_nb:6.3f}, "
                f"reads with no broken chain {z:6.1%}")
    log(f"[chains] -> {path}" + (f", {fig}" if fig else ""))
    return rows


def ground_states(out: Path, cfg: CampaignConfig, log=print) -> list[dict]:
    """The ground state (best configuration) of every instance, from every source available.

    Candidates: the classical reference (exact Gray-code enumeration for N <= n_exact: certified;
    otherwise the best classical solver / long parallel tempering: best known) and, for every
    annealing back-end, its best read and its best polished read. The best candidate is reported
    with its reward R_N, energy H = -N (R_N - offset)/2, configuration (+/- per agent, sign fixed
    so that agent 0 is +, since S and -S are equivalent), where it came from, whether it is
    certified, and for each source whether it found it and how far (Hamming distance) its best
    configuration is. Writes summary/ground_states.csv."""
    labels = sorted(p.name for p in (out / "anneal").glob("*")) if (out / "anneal").exists() else []
    rows = []
    for key, eps, J0 in _classes(cfg):
        for N in cfg.Ns:
            cp = classical_path(out, key, N)
            c = json.loads(cp.read_text()) if cp.exists() else None
            if c is None and not any(anneal_path(out, l, key, N, 0).exists() for l in labels):
                continue
            J = ensure_instances(out, cfg, key, eps, J0, N)
            for b in range(cfg.n_instances):
                P = PlannerProblem(J[b])
                cand = []                                    # (reward, spins, column, source)
                if c is not None and c.get("ref_S"):
                    cand.append((c["ref"][b], c["ref_S"][b], "classical", c["ref_source"][b]))
                for lab in labels:
                    ap = anneal_path(out, lab, key, N, b)
                    if ap.exists():
                        r = json.loads(ap.read_text())
                        if r.get("best_S"):
                            cand.append((r["best_reward"], r["best_S"], lab, lab))
                        if r.get("polished_best_S"):
                            cand.append((r["polished_best_reward"], r["polished_best_S"], lab + "+polish", lab + "+polish"))
                if not cand:
                    continue
                top = max(t[0] for t in cand)                # ties (to rounding) go to the first source: classical/exact
                best = next(t for t in cand if t[0] >= top - TOL * max(1, abs(top)))
                g = _canonical(best[1])
                S = np.array([1.0 if ch == "+" else -1.0 for ch in g])
                certified = bool(c is not None and c.get("ref_exact") and c.get("ref_S"))
                row = {"problem": key, "N": N, "instance": b, "reward": float(P.reward(S)),
                       "energy": float(P.energy(S)), "ground_state": g, "found_by": best[3],
                       "certified": certified, "status": "exact optimum" if certified else "best known"}
                for rw, sp, col, _src in cand:
                    row[f"{col}:found"] = bool(rw >= top - TOL * max(1, abs(top)))
                    row[f"{col}:hamming"] = sum(x != y for x, y in zip(_canonical(sp), g))
                rows.append(row)
    if rows:
        path = out / "summary" / "ground_states.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            log(f"[ground] {r['problem']:10s} N={r['N']:4d} i={r['instance']} R_N={r['reward']:.5f} "
                f"H={r['energy']:9.4f} {r['status']:13s} by {r['found_by']:24s} {r['ground_state'] if r['N'] <= 40 else r['ground_state'][:40] + '...'}")
        log(f"[ground] -> {path}")
    else:
        log("[ground] no results yet")
    return rows


def validate(out: Path, cfg: CampaignConfig, log=print) -> dict:
    """Re-derive every stored number from the stored inputs, a posteriori.

    For each annealer result: rebuild the instance from instances/*.npz and recompute the reward of
    every stored configuration (reads.S, best_S, polished_best_S); check read counts, the broken-
    chain histogram against D-Wave's chain_break_fraction, the chain strength against
    p * J_rms * sqrt(N-1), the embedding_used (valid minor of K_N on the ideal graph of the
    topology, which contains every working graph) and its chain lengths. For each classical
    file: recompute the reward of the reference configuration and, for N <= n_exact, re-run the
    exact enumeration. Returns {check: [passed, failed]}."""
    from skgame.optim.embedding import is_valid_embedding, topology as topo
    from skgame.optim.backends import parse_spec
    tally = {}

    def check(name, ok, detail=""):
        t = tally.setdefault(name, [0, 0])
        t[0 if ok else 1] += 1
        if not ok:
            log(f"[validate] FAILED {name}: {detail}")

    def close(a, b):
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))

    graphs = {}
    for key, eps, J0 in _classes(cfg):
        for N in cfg.Ns:
            if not instance_path(out, key, N).exists():
                continue
            J = ensure_instances(out, cfg, key, eps, J0, N)
            cp = classical_path(out, key, N)
            if cp.exists():
                c = json.loads(cp.read_text())
                for b in range(cfg.n_instances):
                    P = PlannerProblem(J[b])
                    if c.get("ref_S"):
                        S = np.array([1.0 if ch == "+" else -1.0 for ch in c["ref_S"][b]])
                        check("classical reference configuration -> reference value", close(P.reward(S), c["ref"][b]),
                              f"{key} N={N} i={b}")
                    if c["ref_exact"] and N <= 24:
                        check("exact optimum re-computed", close(ExactEnumeration(max_N=24).solve(P).reward, c["ref"][b]),
                              f"{key} N={N} i={b}")
            for lab in sorted(p.name for p in (out / "anneal").glob("*")) if (out / "anneal").exists() else []:
                for b in range(cfg.n_instances):
                    ap = anneal_path(out, lab, key, N, b)
                    if not ap.exists():
                        continue
                    r = json.loads(ap.read_text())
                    info, P, where = r["info"], PlannerProblem(J[b]), f"{lab} {key} N={N} i={b}"
                    to_S = lambda st: np.array([1.0 if ch == "+" else -1.0 for ch in st])
                    if r.get("best_S"):
                        check("best configuration -> best reward", close(P.reward(to_S(r["best_S"])), r["best_reward"]), where)
                    if r.get("polished_best_S"):
                        check("polished configuration -> polished reward",
                              close(P.reward(to_S(r["polished_best_S"])), r["polished_best_reward"]), where)
                    t = r.get("reads")
                    if t and t.get("S"):
                        rec = np.atleast_1d(P.reward(np.array([to_S(x) for x in t["S"]]))) if t["S"] else np.zeros(0)
                        check("every stored configuration -> its reward", np.allclose(rec, t["reward"], rtol=1e-9, atol=1e-12), where)
                    if t:
                        check("read counts add up", sum(t["count"]) == info["num_reads"] - info.get("discarded_reads", 0)
                              or sum(t["count"]) == info["num_reads"], where)
                    h = info.get("broken_chains_hist")
                    if h and "chain_break_fraction_mean" in info and not info.get("discarded_reads"):
                        mean = sum(int(k) * v for k, v in h.items()) / max(1, sum(h.values())) / N
                        check("broken-chain counts agree with D-Wave's chain_break_fraction",
                              abs(mean - info["chain_break_fraction_mean"]) < 1e-9, where)
                    if info.get("chain_strength") is not None:
                        q = P.W[np.triu_indices(N, 1)]
                        expect = info["settings"]["chain_strength_prefactor"] * np.sqrt(np.mean(q ** 2)) * np.sqrt(N - 1)
                        check("chain strength = p J_rms sqrt(N-1)", close(info["chain_strength"], expect), where)
                    pf = ap.with_name(info["physical_problem_file"]) if info.get("physical_problem_file") else None
                    if pf is not None and pf.exists() and info.get("embedding_used"):
                        import dimod
                        from dwave.embedding import EmbeddedStructure
                        ph = np.load(pf)
                        emb = {int(k): v for k, v in info["embedding_used"].items()}
                        edges = list(zip(ph["u"].tolist(), ph["v"].tolist()))
                        rebuilt = EmbeddedStructure(edges, emb).embed_bqm(P.to_bqm(), chain_strength=info["chain_strength"],
                                                                          smear_vartype=dimod.SPIN)
                        saved = dimod.BinaryQuadraticModel(dict(zip(ph["h_q"].tolist(), ph["h"].tolist())),
                                                           dict(zip(edges, ph["J"].tolist())), float(ph["offset"]), "SPIN")
                        check("physical problem = embed(instance, embedding_used, chain_strength)",
                              rebuilt.is_almost_equal(saved), where)
                        sc = info["programmed"].get("scaling", {"mode": "auto_scale"})
                        if sc["mode"] == "extended_j_range":
                            own = {q: i for i, ch in emb.items() for q in ch}
                            chain = np.array([abs(v) for (x, y), v in zip(edges, ph["J"]) if own[x] == own[y]])
                            logi = np.array([abs(v) for (x, y), v in zip(edges, ph["J"]) if own[x] != own[y]])
                            expect = max(chain.max() / -sc["ext_lo"], logi.max() if logi.size else 0.0)
                            check("extended-range divisor = max(c/|ext_lo|, max logical |J|)",
                                  close(sc["divisor"], expect), where)
                        else:
                            check("auto_scale divisor = max |J| of the physical problem",
                                  close(float(np.abs(ph["J"]).max()), info["programmed"]["max_abs_J_before_scaling"]), where)
                    if info.get("embedding_used"):
                        emb = {int(k): v for k, v in info["embedding_used"].items()}
                        t = (info.get("solver_properties") or {}).get("topology") or {}
                        tp = t.get("type") or parse_spec(info["backend"])[1]
                        shape = tuple(t["shape"]) if t.get("shape") else topo(tp).shape
                        from skgame.optim.embedding import Topology
                        G = graphs.setdefault((tp, shape), Topology(tp, shape, "").ideal_graph())
                        check("embedding is a valid minor of K_N", is_valid_embedding(emb, N, G), where)
                        check("chain lengths match the embedding", [len(emb[i]) for i in range(N)] == info["chain_lengths"], where)
    for name, (ok, bad) in tally.items():
        log(f"[validate] {'OK  ' if not bad else 'FAIL'} {name}: {ok} passed, {bad} failed")
    if not tally:
        log("[validate] nothing to validate yet")
    return tally


def scan_summary(dirs, log=print) -> list[dict]:
    """Compare campaigns that differ only in the annealing settings (e.g. one directory per chain-
    strength prefactor, same instances). The reference for each instance is the best configuration
    seen in ANY of the directories (plus the classical/exact one if present), so success rates are
    comparable across settings. Writes scan_summary.csv next to the first directory."""
    runs = []
    for d in map(Path, dirs):
        cfg = load_config(d)
        for f in sorted((d / "anneal").glob("*/*/*.json")) if (d / "anneal").exists() else []:
            r = json.loads(f.read_text())
            runs.append((d, cfg, f.parent.parent.name, r))
    ref = {}
    for d, cfg, lab, r in runs:
        k = (r["key"], r["N"], r["instance"])
        ref[k] = max(ref.get(k, -INF), r["best_reward"], r.get("polished_best_reward", -INF))
        cp = classical_path(d, r["key"], r["N"])
        if cp.exists():
            ref[k] = max(ref[k], json.loads(cp.read_text())["ref"][r["instance"]])
    groups = {}
    for d, cfg, lab, r in runs:
        g = groups.setdefault((str(d), cfg.chain_strength_prefactor, cfg.chain_break_method, cfg.annealing_time,
                               cfg.num_reads, lab, r["N"]), {"calls": 0, "reads": 0, "succ": 0, "succ_p": 0,
                                                            "broken": 0.0, "zero": 0, "res": []})
        k = (r["key"], r["N"], r["instance"])
        cnt = np.array(r["counts"])
        g["calls"] += 1
        g["reads"] += cnt.sum()
        g["succ"] += cnt[_success(r["rewards"], ref[k])].sum() if len(cnt) else 0
        if r.get("polished_counts"):
            pc = np.array(r["polished_counts"])
            g["succ_p"] += pc[_success(r["polished_rewards"], ref[k])].sum()
        h = r["info"].get("broken_chains_hist") or {}
        g["broken"] += sum(int(a) * b for a, b in h.items())
        g["zero"] += h.get("0", 0)
        g["res"].append((ref[k] - r["best_reward"]) / abs(ref[k]))
    rows = []
    for (d, pf, cbm, ta, nr, lab, N), g in sorted(groups.items(), key=lambda t: (t[0][5], t[0][6], t[0][1])):
        n = max(1, g["reads"])
        rows.append({"dir": d, "prefactor": pf, "chain_break_method": cbm, "annealing_time": ta, "num_reads": nr,
                     "backend": lab, "N": N, "calls": g["calls"], "p_success": g["succ"] / n,
                     "p_success_polished": g["succ_p"] / n, "mean_broken_chains": g["broken"] / n,
                     "frac_reads_no_break": g["zero"] / n, "residual_best_median": float(np.median(g["res"]))})
        log(f"[scan] {lab:20s} N={N:4d} p={pf:<6g} {cbm:15s} ta={ta:<5g} P(succ)={rows[-1]['p_success']:.3f} "
            f"polished={rows[-1]['p_success_polished']:.3f} broken/read={rows[-1]['mean_broken_chains']:.3f} "
            f"no-break={rows[-1]['frac_reads_no_break']:.1%}")
    if rows:
        path = Path(dirs[0]).parent / "scan_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"[scan] -> {path}")
    return rows


def stage_analyze(out: Path, cfg: CampaignConfig, log=print, show=False) -> list[dict]:
    report_chain_breaks(out, cfg, log=log)                # works before the classical stage too
    ground_states(out, cfg, log=lambda *a: None)
    rows = collect(out, cfg)
    if not rows:
        log("[analyze] nothing else to analyse yet: run the classical stage first")
        return rows
    path = write_summary(out, rows)
    figs = plot_summary(out, rows, show=show)
    comp = compare_backends(rows)
    if comp:
        cpath = out / "summary" / "backend_comparison.csv"
        cols = []
        for r in comp:
            cols += [k for k in r if k not in cols]
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(comp)
        figs.append(plot_backend_comparison(out, comp, rows, show=show))
        wins = {}
        for r in comp:
            wins[r["winner"]] = wins.get(r["winner"], 0) + 1
        log(f"[analyze] back-end comparison -> {cpath}")
        log("[analyze] points won (problem class x N): " + ", ".join(f"{k}: {v}" for k, v in sorted(wins.items())))
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
    qpu = [b for b in cfg.backends if b.startswith("qpu:")]
    for key, tab in (cfg.chain_strength_by_class or {"(all classes)": cfg.chain_strength_by_N}).items():
        for lab, t in (tab or {}).items():
            log(f"  chain strength prefactor p_opt for {lab} {key} by N (calibration): " +
                ", ".join(f"{k}: {v:g}" for k, v in sorted(t.items(), key=lambda kv: int(kv[0]))))
    log(f"  annealing back-ends: {cfg.backends}  (chain strength prefactor {cfg.chain_strength_prefactor}, "
        f"chain breaks: {cfg.chain_break_method}, polish: {cfg.polish})")
    n_call = len(cfg.Ns) * len(classes) * cfg.n_instances
    # t_call ~ t_programming + num_reads (t_anneal + t_readout + t_delay), Advantage typical values
    t_prog, t_ro, t_d = 15e-3, 100e-6, 20e-6
    t_read = cfg.annealing_time * 1e-6 + t_ro + t_d
    t_call = t_prog + cfg.num_reads * t_read
    log(f"  annealer calls per back-end: |Ns| * classes * instances = {len(cfg.Ns)} * {len(classes)} * "
        f"{cfg.n_instances} = {n_call}; QPU back-ends: {len(qpu)} -> {n_call * len(qpu)} calls")
    log(f"  QPU time per call ~ {t_prog * 1e3:.0f} ms + {cfg.num_reads} * ({cfg.annealing_time:g} + "
        f"{t_ro * 1e6:.0f} + {t_d * 1e6:.0f}) us = {t_call * 1e3:.0f} ms")
    log(f"  QPU access time ~ {len(qpu)} * {n_call} * {t_call:.3f} s = {len(qpu) * n_call * t_call:.0f} s  "
        f"(upper bound; `qpu --check` gives each solver's own estimate)")


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
    p.add_argument("--backends", nargs="+", help="annealing back-ends, e.g. qpu:zephyr:clique qpu:pegasus:clique")
    p.add_argument("--chain-strength-prefactor", type=float, help="x uniform torque compensation (default 1.414)")
    p.add_argument("--chain-break-method", choices=["majority_vote", "minimize_energy", "weighted_random", "discard"])
    p.add_argument("--extended-j-range", action="store_true",
                   help="put chain couplers in the QPU's extended J range (e.g. down to -2): less compression")
    p.add_argument("--force", action="store_true", help="overwrite an existing config.json")
    sub.add_parser("plan", parents=[common], help="print sizes, tasks and the QPU budget")
    sub.add_parser("instances", parents=[common], help="write every instance J now (no internet, seconds)")
    p = sub.add_parser("classical", parents=[common], help="references, classical solvers and the game")
    p.add_argument("--task", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID") and
                   int(os.environ["SLURM_ARRAY_TASK_ID"]), help="problem-class index (default: Slurm array id)")
    p.add_argument("--problem", nargs="+", help="problem-class keys (see `plan`)")
    for name, hlp in [("qpu", "every qpu:* back-end of the config (needs a Leap token and outbound HTTPS)"),
                      ("sim", "classical simulated-annealing stand-in, default sim:logical (no token)")]:
        p = sub.add_parser(name, parents=[common], help=f"annealer stage: {hlp}")
        p.add_argument("--backend", nargs="+", help="back-end spec(s), e.g. qpu:zephyr:clique or sim:pegasus:clique")
        p.add_argument("--problem", nargs="+")
        p.add_argument("--max-calls", type=int, help="stop after this many sampler calls (per back-end)")
        p.add_argument("--solver", help="pin a QPU by name, e.g. Advantage2_system1.4 (only with one --backend)")
        p.add_argument("--Ns", nargs="+", type=parse_Ns, help="only these N, e.g. 8:96:8 or 104")
        if name == "qpu":
            p.add_argument("--check", action="store_true",
                           help="connect, print solver and clique size, estimate QPU time; spends nothing")
            p.add_argument("--budget-seconds", type=float,
                           help="hard cap on the QPU access time this command may spend (all back-ends together)")
            p.add_argument("--spent", action="store_true",
                           help="print the QPU time already spent in this campaign; spends nothing")
    p = sub.add_parser("embed", parents=[common], help="compute and cache embeddings, print chain statistics")
    p.add_argument("--backend", nargs="+", help="default: every back-end of the config (qpu ones need the token)")
    p.add_argument("--solver")
    p = sub.add_parser("analyze", parents=[common], help="summary.csv and figures")
    p = sub.add_parser("coupling-scan", parents=[common],
                       help="choose the chain-strength prefactor for every N by testing it on the QPU")
    p.add_argument("--Ns", nargs="+", type=parse_Ns, default=None, help="N values, e.g. 8:48:8 104")
    p.add_argument("--prefactors", nargs="+", type=float, default=[0.75, 1.0, 1.414, 2.0, 3.0, 4.0, 6.0])
    p.add_argument("--n-instances", type=int, default=1)
    p.add_argument("--num-reads", type=int, default=200)
    p.add_argument("--problem", nargs="+", help="problem classes to scan (default: eps0_J00)")
    p.add_argument("--backend", nargs="+", help="default: the qpu back-ends of the config")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="accepted mean fraction of broken chains (default 0: no chain broken)")
    p.add_argument("--budget-seconds", type=float, help="hard cap on the QPU time of this scan")
    p.add_argument("--check", action="store_true", help="estimate the QPU time; spends nothing")
    p.add_argument("--summary-only", action="store_true", help="only (re)compute the choice from saved results")
    p.add_argument("--no-apply", action="store_true", help="do not store the chosen values in config.json")
    p = sub.add_parser("shim", parents=[common], help="flux-bias shimming of a QPU on the SK problem (tutorial / Teza et al.)")
    p.add_argument("--backend", required=True, help="e.g. qpu:zephyr:clique")
    p.add_argument("--N", type=int, required=True, help="problem size whose embedding is shimmed")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--alpha-phi", type=float, default=1e-5, help="learning rate (Phi0 per unit magnetisation); 0 = diagnose only")
    p.add_argument("--num-reads", type=int, default=100)
    p.add_argument("--annealing-time", type=float)
    p.add_argument("--level", choices=["qubit", "chain"], default="qubit")
    p.add_argument("--no-gauge", action="store_true", help="no random spin-reversal gauge per iteration")
    p.add_argument("--budget-seconds", type=float)
    p.add_argument("--check", action="store_true", help="estimate the QPU time; spends nothing")
    p.add_argument("--no-apply", action="store_true", help="do not use the offsets in later qpu calls")
    # ---- the linear QPU pipeline (pipeline.py): budget -> calibrate -> run -> report
    p = sub.add_parser("budget", parents=[common], help="PIPELINE 0: QPU time of calibrate + run (spends nothing)")
    p.add_argument("--total", type=float, default=178.0,
                   help="QPU seconds you had when this campaign started (default 178); keep it fixed between steps")
    p.add_argument("--fill", action="store_true",
                   help="set num_reads so that the run spends everything left (stored in config.json)")
    p.add_argument("--margin", type=float, default=3.0, help="seconds kept unspent by --fill (default 3)")
    p = sub.add_parser("calibrate", parents=[common], help="PIPELINE 1: p_opt for every (N, machine) of the config")
    p.add_argument("--budget-seconds", type=float)
    p.add_argument("--threshold", type=float, default=0.0, help="accepted fraction of broken chains (default 0)")
    p.add_argument("--problem", nargs="+", help="only these problem classes (default: all of the config)")
    p.add_argument("--check", action="store_true", help="estimate only; spends nothing")
    p = sub.add_parser("run", parents=[common], help="PIPELINE 2: anneal every N with its p_opt, then report")
    p.add_argument("--budget-seconds", type=float)
    p.add_argument("--check", action="store_true", help="estimate only; spends nothing")
    p.add_argument("--allow-uncalibrated", action="store_true", help="interpolate p for N without calibration")
    p.add_argument("--total", type=float,
                   help="QPU seconds available when the campaign started: caps this run at total - already spent")
    sub.add_parser("report", parents=[common], help="PIPELINE 3: ground state, energy, reference per (machine, N)")
    p = sub.add_parser("scan-summary", help="compare campaigns differing only in annealing settings (no QPU)")
    p.add_argument("--dirs", nargs="+", required=True, type=Path)
    sub.add_parser("validate", parents=[common],
                   help="recompute every stored number from the stored inputs (no QPU)")
    sub.add_parser("ground", parents=[common],
                   help="ground state (best configuration) of every instance, its source, certified or not")
    sub.add_parser("chains", parents=[common],
                   help="broken chains per read and solution quality vs number of broken chains (no QPU)")
    a = ap.parse_args(argv)

    if a.cmd == "scan-summary":
        scan_summary(a.dirs)
        return
    out = a.out
    if a.cmd == "init":
        if (out / "config.json").exists() and not a.force:
            cfg = load_config(out)
            print(f"{out}/config.json exists (preset '{cfg.preset}'); use --force to replace it")
        else:
            cfg = make_config(a.preset, Ns=a.Ns, regimes=a.regimes, n_instances=a.n_instances,
                              num_reads=a.num_reads, annealing_time=a.annealing_time, backends=a.backends,
                              chain_strength_prefactor=a.chain_strength_prefactor,
                              chain_break_method=a.chain_break_method,
                              extended_j_range=True if a.extended_j_range else None)
            regimes_by_name(cfg.regimes)
            from skgame.optim.backends import parse_spec
            for spec in cfg.backends:
                parse_spec(spec)
            save_config(out, cfg)
        describe_plan(out, cfg)
        return
    cfg = load_config(out)
    if a.cmd == "plan":
        describe_plan(out, cfg)
    elif a.cmd == "instances":
        for key, eps, J0 in _classes(cfg):
            for N in cfg.Ns:
                ensure_instances(out, cfg, key, eps, J0, N)
        print(f"instances in {out / 'instances'}")
    elif a.cmd == "classical":
        stage_classical(out, cfg, _task_keys(cfg, a.task, a.problem))
    elif a.cmd in ("qpu", "sim"):
        specs = a.backend or ([b for b in cfg.backends if b.startswith("qpu:")] if a.cmd == "qpu" else ["sim:logical"])
        if any(not sp.startswith(a.cmd + ":") for sp in specs):
            sys.exit(f"`{a.cmd}` runs {a.cmd}:* back-ends only, got {specs}")
        if a.solver and len(specs) > 1:
            sys.exit("--solver pins one QPU: pass a single --backend with it")
        if getattr(a, "spent", False):
            qpu_spent(out)
            return
        budget = QPUBudget(getattr(a, "budget_seconds", None))
        for spec in specs:
            Ns = sorted({n for g in a.Ns for n in g}) if a.Ns else None
            stage_anneal(out, cfg, spec, keys=a.problem, max_calls=a.max_calls, solver_name=a.solver,
                         check_only=getattr(a, "check", False), budget=budget, Ns=Ns)
        if budget.limit is not None and not getattr(a, "check", False):
            print(f"[budget] spent {budget.spent:.3f}s of the {budget.limit:.3f}s allowed by --budget-seconds")
    elif a.cmd == "embed":
        stage_embed(out, cfg, a.backend or cfg.backends, solver_name=a.solver)
    elif a.cmd == "analyze":
        stage_analyze(out, cfg)
    elif a.cmd == "chains":
        report_chain_breaks(out, cfg)
    elif a.cmd == "ground":
        ground_states(out, cfg)
    elif a.cmd == "validate":
        validate(out, cfg)
    elif a.cmd in ("budget", "calibrate", "run", "report"):
        import pipeline
        if a.cmd == "budget":
            pipeline.budget(out, cfg, total=a.total, fill=a.fill, margin=a.margin)
        elif a.cmd == "calibrate":
            pipeline.calibrate(out, cfg, budget_s=a.budget_seconds, check=a.check, threshold=a.threshold, keys=a.problem)
        elif a.cmd == "run":
            bs = a.budget_seconds
            if a.total is not None:
                left = a.total - sum(pipeline.spent_in_campaign(out).values())
                print(f"[run] --total {a.total:g}: {left:.2f} s left for this run")
                bs = left if bs is None else min(bs, left)
            pipeline.run(out, cfg, budget_s=bs, check=a.check, allow_uncalibrated=a.allow_uncalibrated)
        else:
            pipeline.report(out, cfg)
    elif a.cmd == "shim":
        import shimming
        shimming.run_shimming(out, cfg, a.backend, a.N, iterations=a.iterations, alpha_phi=a.alpha_phi,
                              num_reads=a.num_reads, annealing_time=a.annealing_time, level=a.level,
                              gauge=not a.no_gauge, budget=QPUBudget(a.budget_seconds), check_only=a.check,
                              apply=not a.no_apply)
    elif a.cmd == "coupling-scan":
        import coupling_scan as cs
        if a.summary_only:
            cs.summarize_coupling_scan(out, cfg, threshold=a.threshold, apply=not a.no_apply)
        else:
            Ns = sorted({n for group in (a.Ns or [[n for n in cfg.Ns if n <= 48]]) for n in group})
            cs.find_best_coupling_per_problem(out, cfg, Ns, prefactors=a.prefactors, specs=a.backend, keys=a.problem,
                                              n_instances=a.n_instances, num_reads=a.num_reads,
                                              threshold=a.threshold, budget=QPUBudget(a.budget_seconds),
                                              check_only=a.check, apply=not a.no_apply)


if __name__ == "__main__":
    main()
