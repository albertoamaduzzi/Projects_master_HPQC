"""The QPU pipeline in four linear steps: list of N -> p_opt(N, machine) -> SK runs -> ground states.

    python main_simulation.py init      --preset full --out results/sk --Ns 8:160:8
    python main_simulation.py budget    --out results/sk --total 178        # 0. what it will cost (spends nothing)
    python main_simulation.py calibrate --out results/sk --budget-seconds 15  # 1. p_opt for every (N, machine)
    python main_simulation.py run       --out results/sk --budget-seconds 160 # 2. the SK problems, with p_opt
    python main_simulation.py report    --out results/sk                    # 3. ground state, energy, ... (no QPU)

Step 1, calibration. For every machine and every N of the list (up to the largest clique the machine
embeds) one SK instance is annealed with p = 0.75, 1, 1.414, 2, 3, 4, 6, climbing from the smallest,
and stops at the first p for which no chain breaks in any read. That p is p_opt(N, machine):
the smallest chain coupling that makes every chain act as one spin. Result: calibration/p_opt.csv,
calibration/p_opt.png; the table is stored in config.json and step 2 uses exactly it.

Step 2, run. Every instance of every problem class is annealed, N by N (both machines at each N, so a
budget stop leaves complete N), with c = p_opt(N, machine) J_rms sqrt(N-1). Refuses to start if a
(N, machine) has no calibration.

Step 3, report. results/sk_qpu_results.csv: per (machine, problem, N, instance) the p used, the chain
strength, qubits, the best configuration found (the ground-state candidate), its energy H and reward
R_N, after polishing too, the reference (exact for small N, else best known if the classical stage ran)
and whether the QPU reached it, broken chains and QPU time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

GRID = (0.75, 1.0, 1.414, 2.0, 3.0, 4.0, 6.0)
CAL_READS = 200


def _qpu_backends(out, cfg):
    import main_simulation as ms
    return [ms.make_backend(out, cfg, s) for s in cfg.backends if s.startswith("qpu:")]


def _pairs(cfg, backends):
    """(backend, N) pairs the machine can embed, N-major."""
    pairs = []
    for N in cfg.Ns:
        for be in backends:
            lim = be.largest_problem()
            if lim is None or N <= lim:
                pairs.append((be, N))
    return pairs


def budget(out: Path, cfg, total: float = 178.0, log=print) -> dict:
    """Predicted QPU time of calibration (full grid: upper bound) and of the run; spends nothing."""
    import main_simulation as ms
    bes = _qpu_backends(out, cfg)
    for be in bes:
        lim = be.largest_problem()
        skip = [N for N in cfg.Ns if lim is not None and N > lim]
        log(f"[budget] {be.spec}: solver {be.solver_id()}, largest N {lim}" + (f"; N={skip} not embeddable" if skip else ""))
    n_cls = len(ms._classes(cfg))
    cal = run = 0.0
    for be, N in _pairs(cfg, bes):
        be.settings.num_reads = CAL_READS
        c1 = ms._call_estimate(be, N, ms.CampaignConfig(**{**cfg.__dict__, "num_reads": CAL_READS}))
        be.settings.num_reads = cfg.num_reads
        c2 = ms._call_estimate(be, N, cfg)
        cal += len(GRID) * c1
        run += n_cls * cfg.n_instances * c2
    log(f"[budget] 1. calibrate: {len(_pairs(cfg, bes))} (N, machine) pairs x {len(GRID)} p x {CAL_READS} reads "
        f"<= {cal:.1f} s (adaptive: usually about half)")
    log(f"[budget] 2. run:       {n_cls} problem classes x {cfg.n_instances} instances x {cfg.num_reads} reads "
        f"per (N, machine) = {run:.1f} s")
    log(f"[budget]    total <= {cal + run:.1f} s of {total:.1f} s available -> "
        + ("fits" if cal + run <= total else
           f"does NOT fit: lower --num-reads or --n-instances in `init`, or shorten --Ns "
           f"(the run scales linearly with reads x instances)"))
    return {"calibrate_s": cal, "run_s": run, "total_s": total}


def calibrate(out: Path, cfg, budget_s: float | None = None, check: bool = False, threshold: float = 0.0,
              log=print) -> dict:
    import main_simulation as ms
    import coupling_scan as cs
    best = cs.find_best_coupling_per_problem(out, cfg, cfg.Ns, prefactors=GRID, num_reads=CAL_READS,
                                             threshold=threshold, budget=ms.QPUBudget(budget_s),
                                             check_only=check, apply=True, adaptive=True, log=log)
    if check or not best:
        return best
    _write_table(out, cfg, best, log)
    return best


def _write_table(out, cfg, best, log):
    rows = []
    scan = out / "coupling_scan" / "coupling_scan.csv"
    info = {}
    if scan.exists():
        for r in csv.DictReader(open(scan)):
            info[(r["backend"], int(r["N"]), float(r["p"]))] = r
    for lab, t in best.items():
        for N in sorted(int(k) for k in t):
            p = float(t[str(N)])
            r = info.get((lab, N, p), {})
            rows.append({"machine": lab, "N": N, "p_opt": p, "chain_strength": r.get("chain_strength"),
                         "broken_fraction_at_p_opt": r.get("broken_fraction"),
                         "note": r.get("reason", "")})
    d = out / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "p_opt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for lab in best:
        sel = [r for r in rows if r["machine"] == lab]
        ax.plot([r["N"] for r in sel], [r["p_opt"] for r in sel], "o-", label=lab)
    ax.set_xlabel("N (agents)")
    ax.set_ylabel("p_opt (smallest p with no broken chain)")
    ax.set_yticks(GRID)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(d / "p_opt.png", dpi=130)
    plt.close(fig)
    log("[calibrate] p_opt(N, machine):")
    for r in rows:
        log(f"[calibrate]   {r['machine']:20s} N={r['N']:4d}  p_opt={r['p_opt']:<6g} {r['note']}")
    log(f"[calibrate] -> {d / 'p_opt.csv'}, {d / 'p_opt.png'} (and config.json: step 2 uses this table)")


def missing_calibration(out: Path, cfg, backends=None) -> list:
    bes = backends or _qpu_backends(out, cfg)
    t = cfg.chain_strength_by_N or {}
    return [(be.label, N) for be, N in _pairs(cfg, bes) if str(N) not in (t.get(be.label) or {})]


def run(out: Path, cfg, budget_s: float | None = None, check: bool = False, allow_uncalibrated: bool = False,
        log=print) -> None:
    import main_simulation as ms
    bes = _qpu_backends(out, cfg)
    miss = missing_calibration(out, cfg, bes)
    if miss and not allow_uncalibrated:
        raise SystemExit(f"[run] no p_opt for {len(miss)} (machine, N): {miss[:8]}{' ...' if len(miss) > 8 else ''}\n"
                         f"      run `calibrate` first (or --allow-uncalibrated to interpolate)")
    budget = ms.QPUBudget(budget_s)
    for N in cfg.Ns:                     # N-major: both machines at each N before the next N
        for be in bes:
            lim = be.largest_problem()
            if lim is not None and N > lim:
                continue
            ms.stage_anneal(out, cfg, be.spec, backend=be, budget=budget, Ns=[N], check_only=check, log=log)
            if budget.limit is not None and not check and not budget.allows(ms._call_estimate(be, N, cfg)):
                log(f"[run] budget reached ({budget.spent:.2f} s of {budget.limit:.2f} s): rerun to continue")
                return report(out, cfg, log=log) if not check else None
    if not check:
        log(f"[run] done; QPU time of this command {budget.spent:.2f} s")
        report(out, cfg, log=log)


def report(out: Path, cfg, log=print) -> list:
    """One row per (machine, problem, N, instance): p, ground-state candidate, energy, reference."""
    import main_simulation as ms
    from skgame.optim.problem import PlannerProblem
    from skgame.optim.solvers import _exact_max
    labels = sorted(p.name for p in (out / "anneal").glob("qpu_*")) if (out / "anneal").exists() else []
    rows, exact_cache = [], {}
    for key, eps, J0 in ms._classes(cfg):
        for N in cfg.Ns:
            cp = ms.classical_path(out, key, N)
            c = json.loads(cp.read_text()) if cp.exists() else None
            for b in range(cfg.n_instances):
                res = {lab: ms.anneal_path(out, lab, key, N, b) for lab in labels}
                res = {lab: json.loads(p.read_text()) for lab, p in res.items() if p.exists()}
                if not res:
                    continue
                P = PlannerProblem(ms.ensure_instances(out, cfg, key, eps, J0, N)[b])
                if c is not None and c.get("ref") is not None:
                    ref, src = float(c["ref"][b]), c["ref_source"][b]
                elif N <= min(cfg.n_exact, 24):
                    if (key, N, b) not in exact_cache:
                        S, _ = _exact_max(P.W)
                        exact_cache[(key, N, b)] = float(P.reward(np.asarray(S, dtype=float)))
                    ref, src = exact_cache[(key, N, b)], "exact"
                else:
                    ref, src = None, "none (run `classical` for a best-known reference)"
                for lab, r in res.items():
                    info = r["info"]
                    Sb = np.array([1.0 if ch == "+" else -1.0 for ch in r["best_S"]])
                    pol = r.get("polished_best_reward")
                    best_any = max(r["best_reward"], pol if pol is not None else -np.inf)
                    h = info.get("broken_chains_hist") or {}
                    nr = max(1, sum(h.values()))
                    rows.append({
                        "machine": lab, "problem": key, "N": N, "instance": b,
                        "p_used": info.get("chain_strength_prefactor"),
                        "chain_strength": info.get("chain_strength"),
                        "n_qubits": (info.get("embedding") or {}).get("n_qubits"),
                        "energy_H": float(P.energy(Sb)), "reward_R": r["best_reward"],
                        "reward_R_polished": pol,
                        "reference_R": ref, "reference_source": src,
                        "reached_reference": None if ref is None else bool(best_any >= ref - 1e-9 * max(1, abs(ref))),
                        "gap_R": None if ref is None else float(ref - best_any),
                        "frac_reads_no_broken_chain": h.get("0", 0) / nr,
                        "qpu_access_s": info.get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6,
                        "best_configuration": ms._canonical(r["best_S"]),
                    })
    if not rows:
        log("[report] no QPU results yet")
        return rows
    path = out / "results" / "sk_qpu_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"[report] {'machine':20s} {'problem':9s} {'N':>4s} i  {'p':>5s} {'H':>10s} {'R_N':>8s} {'polished':>8s} {'ref R_N':>8s} reached")
    for r in rows:
        ref = "-" if r["reference_R"] is None else f"{r['reference_R']:.5f}"
        log(f"[report] {r['machine']:20s} {r['problem']:9s} {r['N']:4d} {r['instance']}  {'-' if r['p_used'] is None else format(r['p_used'], '5g'):>5s} "
            f"{r['energy_H']:10.4f} {r['reward_R']:8.5f} "
            f"{'-' if r['reward_R_polished'] is None else format(r['reward_R_polished'], '8.5f'):>8s} {ref:>8s} "
            f"{r['reached_reference']}")
    log(f"[report] -> {path}  (configurations: column best_configuration, + / - per agent, agent 0 = +)")
    return rows
