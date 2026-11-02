"""The QPU pipeline in linear steps: list of N -> p_opt(class, N, machine) -> SK runs -> ground states.

    python main_simulation.py init      --preset full --out results/sk --Ns 8:160:8 --n-instances 5
    python main_simulation.py budget    --out results/sk --total 178        # 0. what it will cost (spends nothing)
    python main_simulation.py calibrate --out results/sk --budget-seconds 55  # 1. p_opt for every (class, N, machine)
    python main_simulation.py budget    --out results/sk --total 178 --fill # 1b. reads that spend all that is left
    python main_simulation.py run       --out results/sk --total 178        # 2. the SK problems, with p_opt
    python main_simulation.py report    --out results/sk                    # 3. ground state, energy, ... (no QPU)

Step 1, calibration. For every problem class, every machine and every N of the list (up to the largest clique the machine
embeds) one SK instance is annealed with p = 0.75, 1, 1.414, 2, 3, 4, 6, climbing from the smallest,
and stops at the first p for which no chain breaks in any read. That p is p_opt(N, machine):
the smallest chain coupling that makes every chain act as one spin (per class: the ferromagnetic
class has stronger fields on the chains and needs a larger p). Result: calibration/p_opt.csv,
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


def spent_in_campaign(out: Path) -> dict:
    """QPU access time already charged for this campaign directory, from the saved results."""
    tot = {"calibrate": 0.0, "run": 0.0, "shim": 0.0}
    for part, pat in (("calibrate", "coupling_scan/qpu_*/*/*.json"), ("run", "anneal/qpu_*/*/*.json")):
        for f in out.glob(pat):
            tot[part] += json.loads(f.read_text())["info"].get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
    for f in out.glob("shims/*/history.json"):
        tot["shim"] += sum(h.get("qpu_access_s") or 0.0 for h in json.loads(f.read_text()))
    return tot


def _est(be, N, cfg, reads):
    """Predicted access time of one call with `reads` reads (solver timing model, or the pessimistic
    fallback of main_simulation._call_estimate)."""
    import main_simulation as ms
    old = be.settings.num_reads
    be.settings.num_reads = reads
    try:
        return ms._call_estimate(be, N, ms.CampaignConfig(**{**cfg.__dict__, "num_reads": reads}))
    finally:
        be.settings.num_reads = old


def budget(out: Path, cfg, total: float = 178.0, fill: bool = False, margin: float = 3.0, log=print) -> dict:
    """What calibration and run cost, and (fill=True) the number of reads per call that makes the run
    spend everything left: total - already spent - calibration still to do - margin. Spends nothing.

    `total` = QPU seconds you had when this campaign started (keep the same value at every step:
    what the campaign has already spent is read from its own results and subtracted)."""
    import main_simulation as ms
    import coupling_scan as cs
    bes = _qpu_backends(out, cfg)
    for be in bes:
        lim = be.largest_problem()
        skip = [N for N in cfg.Ns if lim is not None and N > lim]
        log(f"[budget] {be.spec}: solver {be.solver_id()}, largest N {lim}" + (f"; N={skip} not embeddable" if skip else ""))
    classes = ms._classes(cfg)
    spent = spent_in_campaign(out)
    reached = cs._saved_ok(out, 0.0)
    cal_left = run_fixed = run_per_read = run_now = 0.0
    max_call_reads = 10 ** 9
    for be, N in _pairs(cfg, bes):
        e1, e2 = _est(be, N, cfg, 1), _est(be, N, cfg, 1001)
        per_read = (e2 - e1) / 1000
        max_call_reads = min(max_call_reads, int(0.95 / per_read) if per_read > 0 else 10 ** 6)
        for key, *_ in classes:
            if (be.label, key, N) not in reached:
                done = [p for p in GRID if cs.scan_path(out, be.label, key, N, p, 0).exists()]
                cal_left += (len(GRID) - len(done)) * _est(be, N, cfg, CAL_READS)
            todo = sum(not ms.anneal_path(out, be.label, key, N, b).exists() for b in range(cfg.n_instances))
            run_fixed += todo * (e1 - per_read)
            run_per_read += todo * per_read
            run_now += todo * _est(be, N, cfg, cfg.num_reads)
    left = total - sum(spent.values())
    log(f"[budget] available {total:.1f} s; already spent by this campaign: calibration {spent['calibrate']:.1f} s, "
        f"run {spent['run']:.1f} s, shim {spent['shim']:.1f} s -> left {left:.1f} s")
    log(f"[budget] 1. calibrate: {len(classes)} problem classes x {len(_pairs(cfg, bes))} (N, machine) pairs; "
        f"still to do <= {cal_left:.1f} s (full grid; the climb usually stops earlier)")
    log(f"[budget] 2. run:       {len(classes)} classes x {cfg.n_instances} instances x {cfg.num_reads} reads per "
        f"(N, machine); still to do {run_now:.1f} s")
    ok = cal_left + run_now <= left
    log(f"[budget]    calibrate + run = {cal_left + run_now:.1f} s of {left:.1f} s left -> "
        + ("fits" if ok else "does NOT fit (use --fill to size the reads)"))
    res = {"left_s": left, "calibrate_left_s": cal_left, "run_left_s": run_now, "num_reads": cfg.num_reads}
    if fill and run_per_read > 0:
        target = left - cal_left - margin
        reads = int((target - run_fixed) // run_per_read)
        reads = max(0, min(reads, max_call_reads))
        if cal_left > 0:
            log(f"[budget] calibration is not finished: its upper bound {cal_left:.1f} s is reserved. "
                f"Run `budget --fill` again after `calibrate` to give that time to the run.")
        if any(out.glob("anneal/qpu_*/*/*.json")):
            log("[budget] the run has started: --fill changes the reads only of the calls still to do")
        if reads < 50:
            log(f"[budget] only {reads} reads per call would fit: lower --n-instances or shorten --Ns in `init`")
        else:
            cost = run_fixed + reads * run_per_read
            log(f"[budget] --fill: {reads} reads per call -> run {cost:.1f} s, leaving {left - cal_left - cost:.1f} s "
                f"(margin {margin:g} s; max {max_call_reads} reads fit in one call's 1 s limit)")
            cfg.num_reads = reads
            ms.save_config(out, cfg)
            log(f"[budget] num_reads = {reads} stored in {out / 'config.json'}")
            res["num_reads"] = reads
    return res


def calibrate(out: Path, cfg, budget_s: float | None = None, check: bool = False, threshold: float = 0.0,
              keys=None, log=print) -> dict:
    """p_opt for every (problem class, N, machine): adaptive climb of GRID, 200 reads, instance 0."""
    import main_simulation as ms
    import coupling_scan as cs
    keys = keys or [k for k, *_ in ms._classes(cfg)]
    cs.find_best_coupling_per_problem(out, cfg, cfg.Ns, prefactors=GRID, num_reads=CAL_READS, keys=keys,
                                      threshold=threshold, budget=ms.QPUBudget(budget_s),
                                      check_only=check, apply=True, adaptive=True, log=log)
    if check:
        return {}
    cfg2 = ms.load_config(out)
    if cfg2.chain_strength_by_class:
        _write_table(out, cfg2.chain_strength_by_class, log)
    return cfg2.chain_strength_by_class


def _write_table(out, by_class, log):
    rows = []
    scan = out / "coupling_scan" / "coupling_scan.csv"
    info = {}
    if scan.exists():
        for r in csv.DictReader(open(scan)):
            info[(r["backend"], r["problem"], int(r["N"]), float(r["p"]))] = r
    for key, best in by_class.items():
        for lab, t in best.items():
            for N in sorted(int(k) for k in t):
                p = float(t[str(N)])
                r = info.get((lab, key, N, p), {})
                rows.append({"machine": lab, "problem": key, "N": N, "p_opt": p,
                             "chain_strength": r.get("chain_strength"),
                             "broken_fraction_at_p_opt": r.get("broken_fraction"), "note": r.get("reason", "")})
    d = out / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "p_opt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(dict.fromkeys(r["machine"] for r in rows))
    fig, axes = plt.subplots(1, len(labs), figsize=(6 * len(labs), 4), squeeze=False)
    for ax, lab in zip(axes[0], labs):
        for key in by_class:
            sel = [r for r in rows if r["machine"] == lab and r["problem"] == key]
            if sel:
                ax.plot([r["N"] for r in sel], [r["p_opt"] for r in sel], "o-", label=key)
        ax.set_title(lab)
        ax.set_xlabel("N (agents)")
        ax.set_ylabel("p_opt (smallest p with no broken chain)")
        ax.set_yscale("log")
        ax.set_yticks(GRID)
        ax.set_yticklabels([f"{g:g}" for g in GRID])
        ax.legend(title="problem class", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(d / "p_opt.png", dpi=130)
    plt.close(fig)
    log("[calibrate] p_opt(problem class, N, machine):")
    for r in rows:
        log(f"[calibrate]   {r['machine']:20s} {r['problem']:9s} N={r['N']:4d}  p_opt={r['p_opt']:<6g} {r['note']}")
    log(f"[calibrate] -> {d / 'p_opt.csv'}, {d / 'p_opt.png'} (and config.json: step 2 uses this table)")


def missing_calibration(out: Path, cfg, backends=None) -> list:
    import main_simulation as ms
    bes = backends or _qpu_backends(out, cfg)
    t = cfg.chain_strength_by_class or {}
    return [(be.label, key, N) for be, N in _pairs(cfg, bes) for key, *_ in ms._classes(cfg)
            if str(N) not in ((t.get(key) or {}).get(be.label) or {})]


def run(out: Path, cfg, budget_s: float | None = None, check: bool = False, allow_uncalibrated: bool = False,
        log=print) -> None:
    import main_simulation as ms
    bes = _qpu_backends(out, cfg)
    miss = missing_calibration(out, cfg, bes)
    if miss and not allow_uncalibrated:
        raise SystemExit(f"[run] no p_opt for {len(miss)} (machine, problem, N): {miss[:8]}{' ...' if len(miss) > 8 else ''}\n"
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
