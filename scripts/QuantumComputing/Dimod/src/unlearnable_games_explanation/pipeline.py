"""The QPU pipeline in linear steps: list of N -> p_opt(N, machine) -> SK runs -> ground states.

    python main_simulation.py init      --preset full --out results/sk --Ns 8:160:8 --n-instances 5
    python main_simulation.py budget    --out results/sk --total 178        # 0. what it will cost (spends nothing)
    python main_simulation.py calibrate --out results/sk --budget-seconds 55  # 1. p_opt for every (N, machine), all classes
    python main_simulation.py budget    --out results/sk --total 178 --fill # 1b. reads that spend all that is left
    python main_simulation.py run       --out results/sk --total 178        # 2. the SK problems, with p_opt
    python main_simulation.py report    --out results/sk                    # 3. ground state, energy, ... (no QPU)

Step 1, calibration. For every machine and every N of the list, on the pure SK class (up to the largest clique the machine
embeds) one SK instance is annealed with p = 0.75, 1, 1.414, 2, 3, 4, 6, climbing from the smallest,
and stops at the first p for which no chain breaks in any read. That p is p_opt(N, machine):
the smallest chain coupling that makes every chain act as one spin. One value per (machine, N) is
shared by every problem class (the embedding depends only on topology and N; the ferromagnetic class
has stronger fields, so its chains may break a little more often: `report`/`compare` show it). Result: calibration/p_opt.csv,
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
    ckey = calibration_class(cfg, getattr(cfg, "calibration_class", None))
    cal_left = run_fixed = run_per_read = run_now = 0.0
    max_call_reads = 10 ** 9
    for be, N in _pairs(cfg, bes):
        e1, e2 = _est(be, N, cfg, 1), _est(be, N, cfg, 1001)
        per_read = (e2 - e1) / 1000
        max_call_reads = min(max_call_reads, int(0.95 / per_read) if per_read > 0 else 10 ** 6)
        if (be.label, ckey, N) not in reached and str(N) not in ((cfg.chain_strength_by_N or {}).get(be.label) or {}):
            done = [p for p in GRID if cs.scan_path(out, be.label, ckey, N, p, 0).exists()]
            cal_left += (len(GRID) - len(done)) * _est(be, N, cfg, CAL_READS)
        for key, *_ in classes:
            todo = sum(not ms.anneal_path(out, be.label, key, N, b).exists() for b in range(cfg.n_instances))
            run_fixed += todo * (e1 - per_read)
            run_per_read += todo * per_read
            run_now += todo * _est(be, N, cfg, cfg.num_reads)
    left = total - sum(spent.values())
    log(f"[budget] available {total:.1f} s; already spent by this campaign: calibration {spent['calibrate']:.1f} s, "
        f"run {spent['run']:.1f} s, shim {spent['shim']:.1f} s -> left {left:.1f} s")
    log(f"[budget] 1. calibrate: {len(_pairs(cfg, bes))} (N, machine) pairs, shared by the {len(classes)} classes; "
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


def calibration_class(cfg, key=None) -> str:
    """The problem class the shared calibration is measured on: `key`, else the pure SK class, else the first."""
    import main_simulation as ms
    keys = [k for k, *_ in ms._classes(cfg)]
    if key is not None:
        if key not in keys:
            raise SystemExit(f"[calibrate] {key} is not a problem class of this campaign: {keys}")
        return key
    return "eps0_J00" if "eps0_J00" in keys else keys[0]


def calibrate(out: Path, cfg, budget_s: float | None = None, check: bool = False, threshold: float = 0.0,
              key=None, log=print) -> dict:
    """ONE p_opt per (machine, N), shared by every problem class: the embedding depends only on the
    topology and N. Measured on one class (default the pure SK one): adaptive climb of GRID, 200 reads,
    instance 0. Stored in config.json as chain_strength_by_N; any per-class table is removed."""
    import main_simulation as ms
    import coupling_scan as cs
    key = calibration_class(cfg, key)
    log(f"[calibrate] shared calibration per (machine, N), measured on problem class {key}")
    cs.find_best_coupling_per_problem(out, cfg, cfg.Ns, prefactors=GRID, num_reads=CAL_READS, keys=[key],
                                      threshold=threshold, budget=ms.QPUBudget(budget_s),
                                      check_only=check, apply=True, adaptive=True, log=log)
    if check:
        return {}
    cfg2 = ms.load_config(out)
    best = (cfg2.chain_strength_by_class or {}).get(key) or {}
    cfg2.chain_strength_by_N = best
    cfg2.chain_strength_by_class = {}
    cfg2.calibration_class = key
    ms.save_config(out, cfg2)
    if best:
        _write_table(out, {key: best}, log)
    return best


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
    log(f"[calibrate] p_opt(N, machine), used by EVERY problem class (measured on {next(iter(by_class))}):")
    for r in rows:
        log(f"[calibrate]   {r['machine']:20s} {r['problem']:9s} N={r['N']:4d}  p_opt={r['p_opt']:<6g} {r['note']}")
    log(f"[calibrate] -> {d / 'p_opt.csv'}, {d / 'p_opt.png'} (and config.json: step 2 uses this table)")


def missing_calibration(out: Path, cfg, backends=None) -> list:
    import main_simulation as ms
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


# ================================================================== 4. compare =========
PARISI_E = -0.7633            # SK ground-state energy per spin, N -> infinity (J_ij ~ N(0, 1/N))
FINITE_SIZE_A = 0.7           # e0(N) ~ e_inf + A N^(-2/3)  (finite-size correction of the SK ground state)


def _ttt(p: float, t: float) -> float:
    """Time to reach the target at least once with 99 % probability, from p per sample and t per sample."""
    if not np.isfinite(t) or p <= 0:
        return float("inf")
    if p >= 1:
        return float(t)
    return float(t * np.log(0.01) / np.log1p(-p))


def _samples_stats(rhos: list, weights: list, deltas) -> dict:
    """rhos/weights: one array per instance (residual of every sample, its multiplicity)."""
    best = [float(np.min(r)) for r in rhos]
    typ = []
    for r, w in zip(rhos, weights):
        o = np.argsort(r)
        cw = np.cumsum(np.asarray(w, float)[o])
        typ.append(float(np.asarray(r)[o][np.searchsorted(cw, cw[-1] / 2)]))
    d = {"rho_best_median": float(np.median(best)), "rho_typical_median": float(np.median(typ))}
    for dl in deltas:
        d[f"P_rho_le_{dl:g}"] = float(np.mean([np.sum(np.asarray(w)[np.asarray(r) <= dl + 1e-12]) / np.sum(w)
                                              for r, w in zip(rhos, weights)]))
    return d


def compare(out: Path, cfg, deltas=(0.0, 0.01, 0.02, 0.05), log=print) -> list:
    """The central planner on the QPU vs classical planners vs no planner (the game), per problem class
    (easy -> hard) and N. No QPU time.

    Reference per instance: the exact optimum for N <= n_exact (computed here if the classical stage
    has not run), above that the BEST KNOWN reward over every method (flagged: success against it is
    a lower bound on difficulty, not a certified optimum).
    For every method a "sample" is one QPU read, one run of a classical solver, or one run of the game.
    Residual rho = (R_ref - R) / |R_ref|. Metrics:
      rho_best_median     best sample per instance (median over instances)
      rho_typical_median  median sample per instance
      P_rho_le_d          fraction of samples within d of the reference (d = 0: exact success;
                          d > 0: "satisficing" success, meaningful at every N)
      TTT99_d             time to reach rho <= d once with 99 % probability (QPU: anneal + readout +
                          delay per read; *_incl_programming: whole access time / reads; classical: wall
                          time of one run; game: wall time of one run)
    Writes compare/compare.csv, compare/compare.png, compare/reference_quality.png."""
    import main_simulation as ms
    from skgame.optim.problem import PlannerProblem
    from skgame.optim.regimes import j0_eff, planner_phase, regimes_by_name
    from skgame.optim.solvers import _exact_max
    labels = sorted(p.name for p in (out / "anneal").glob("*")) if (out / "anneal").exists() else []
    regimes = regimes_by_name(cfg.regimes)
    classes = sorted(ms._classes(cfg), key=lambda c: -j0_eff(c[1], c[2]))          # easy (ferro) -> hard
    rows, refq = [], []
    for key, eps, J0 in classes:
        for N in cfg.Ns:
            cp = ms.classical_path(out, key, N)
            c = json.loads(cp.read_text()) if cp.exists() else None
            ann = {lab: [ms.anneal_path(out, lab, key, N, b) for b in range(cfg.n_instances)] for lab in labels}
            ann = {lab: [json.loads(p.read_text()) if p.exists() else None for p in ps] for lab, ps in ann.items()}
            ann = {lab: rs for lab, rs in ann.items() if any(r is not None for r in rs)}
            games = {r.name: json.loads(ms.game_path(out, r.name, N).read_text()) for r in regimes
                     if r.problem_key == key and ms.game_path(out, r.name, N).exists()}
            if c is None and not ann:
                continue
            J = ms.ensure_instances(out, cfg, key, eps, J0, N)
            ref, kind = [], []
            for b in range(cfg.n_instances):
                cand = []
                if c is not None:
                    cand.append(float(c["ref"][b]))
                    for v in c["solvers"].values():
                        cand += [float(x) for x in v["reward"][b]]
                for rs in ann.values():
                    if rs[b] is not None:
                        cand.append(float(rs[b]["best_reward"]))
                        if rs[b].get("polished_best_reward") is not None:
                            cand.append(float(rs[b]["polished_best_reward"]))
                for g in games.values():
                    cand.append(float(np.max(g["best_reward"][b])))
                if c is not None and c.get("ref_exact"):
                    ref.append(float(c["ref"][b])); kind.append("exact")
                elif N <= min(cfg.n_exact, 24):
                    P = PlannerProblem(J[b]); S, _ = _exact_max(P.W)
                    ref.append(float(P.reward(np.asarray(S, float)))); kind.append("exact")
                else:
                    ref.append(max(cand)); kind.append("best_known")
            ref = np.array(ref)
            base = {"problem": key, "eps": eps, "J0": J0, "planner_phase": planner_phase(eps, J0), "N": N,
                    "reference": "exact" if all(k == "exact" for k in kind) else "best_known",
                    "R_ref_mean": float(ref.mean())}
            refq.append(dict(base))

            def add(method, kind_, rhos, weights, t_sample, extra=None):
                d = dict(base, method=method, kind=kind_, n_instances=len(rhos), t_sample_s=t_sample)
                d.update(_samples_stats(rhos, weights, deltas))
                for dl in deltas:
                    d[f"TTT99_{dl:g}_s"] = _ttt(d[f"P_rho_le_{dl:g}"], t_sample)
                d.update(extra or {})
                rows.append(d)

            for lab, rs in ann.items():
                done = [(b, r) for b, r in enumerate(rs) if r is not None]
                rho = [(ref[b] - np.array(r["rewards"])) / abs(ref[b]) for b, r in done]
                w = [np.array(r["counts"], float) for b, r in done]
                t_read = float(np.median([ms._time_per_read(r) for b, r in done]))
                acc = [r["info"].get("timing_us", {}).get("qpu_access_time", np.nan) / 1e6 / r["info"]["num_reads"]
                       for b, r in done]
                t_prog = float(np.nanmedian(acc)) if np.isfinite(acc).any() else float("nan")
                pu = [r["info"].get("chain_strength_prefactor") for b, r in done]
                extra = {"p_used": pu[0], "t_sample_incl_programming_s": t_prog,
                         "frac_reads_no_broken_chain": float(np.mean([(r["info"].get("broken_chains_hist") or {}).get("0", 0)
                                                                      / max(1, r["info"]["num_reads"]) for b, r in done]))}
                add(lab, "qpu" if lab.startswith("qpu_") else "anneal_sim", rho, w, t_read, extra)
                rows[-1].update({f"TTT99_{dl:g}_incl_programming_s": _ttt(rows[-1][f"P_rho_le_{dl:g}"], t_prog)
                                 for dl in deltas})
                if all(r.get("polished_rewards") for b, r in done):
                    rho = [(ref[b] - np.array(r["polished_rewards"])) / abs(ref[b]) for b, r in done]
                    w = [np.array(r["polished_counts"], float) for b, r in done]
                    add(lab + "+polish", "qpu" if lab.startswith("qpu_") else "anneal_sim", rho, w, float("nan"),
                        {"p_used": pu[0]})       # polishing runs on the CPU: no fair per-sample time, no TTT
            if c is not None:
                for name, v in c["solvers"].items():
                    R = np.array(v["reward"], float)
                    rho = [(ref[b] - R[b]) / abs(ref[b]) for b in range(len(ref))]
                    add(name, "classical", rho, [np.ones(R.shape[1])] * len(ref), float(np.median(v["seconds"])))
            for reg, g in games.items():
                Rf = np.array(g["final_reward"], float)
                rho = [(ref[b] - Rf[b]) / abs(ref[b]) for b in range(len(ref))]
                L = np.array(g["period"])
                add(f"game:{reg}", "game", rho, [np.ones(Rf.shape[1])] * len(ref),
                    float(g["seconds"]) / Rf.size,
                    {"frac_fixed_point": float(np.mean(L == 1)), "frac_cycle": float(np.mean(L > 1))})
    if not rows:
        log("[compare] nothing to compare yet")
        return rows
    d = out / "compare"
    d.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(d / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _plot_compare(d, rows, [c[0] for c in classes], deltas, cfg.n_exact)
    _plot_reference(d, refq, classes)
    for r in rows:
        if r["kind"] in ("qpu", "game") or r["method"] in ("pt", "sa", "parallel_tempering"):
            log(f"[compare] {r['problem']:9s} N={r['N']:4d} {r['method']:28s} rho_best={r['rho_best_median']:.4f} "
                f"P(rho<=1%)={r.get('P_rho_le_0.01', float('nan')):.3f} P(exact)={r['P_rho_le_0']:.3f} ({r['reference']})")
    log(f"[compare] -> {d / 'compare.csv'}, {d / 'compare.png'}, {d / 'reference_quality.png'}")
    return rows


def _plot_compare(d: Path, rows: list, keys: list, deltas, n_exact: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in keys if any(r["problem"] == k for r in rows)]
    sat = next((dl for dl in deltas if dl > 0), None)
    panels = [("rho_best_median", "best sample: residual (R_ref-R)/R_ref\n(1e-5 = reached it)", True),
              ("rho_typical_median", "typical sample: residual\n(1e-5 = reached it)", True),
              ("P_rho_le_0", "P(exact reference) per sample", False)]
    if sat is not None:
        panels += [(f"P_rho_le_{sat:g}", f"P(residual <= {sat:g}) per sample\n(satisficing success)", False),
                   (f"TTT99_{sat:g}_s", f"time to residual <= {sat:g}, 99% [s]", True)]
    fig, axes = plt.subplots(len(panels), len(keys), figsize=(4.4 * len(keys), 2.9 * len(panels)), squeeze=False)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    kind_of = {r["method"]: r["kind"] for r in rows}
    qcol = {"zephyr": "#1f4fbf", "pegasus": "#c0392b"}
    ccols = plt.get_cmap("tab10").colors[2:] + plt.get_cmap("Dark2").colors
    cl = [m for m in methods if kind_of[m] == "classical"]
    gm = [m for m in methods if kind_of[m] == "game"]

    def style(m):
        k = kind_of[m]
        if k in ("qpu", "anneal_sim"):
            col = next((c for t, c in qcol.items() if t in m), "#555555")
            pol = m.endswith("+polish")
            return dict(color=col, lw=2.6 if not pol else 1.6, ls="-" if not pol else "-.",
                        marker="o" if not pol else "^", ms=4, alpha=1.0 if k == "qpu" else 0.5)
        if k == "game":
            return dict(color="black", lw=1.5, ls="--", marker="sDv^"[gm.index(m) % 4], ms=4)
        return dict(color=ccols[cl.index(m) % len(ccols)], lw=0.9, alpha=0.75, marker=".")
    for j, key in enumerate(keys):
        sub = [r for r in rows if r["problem"] == key]
        for i, (m, lab, logy) in enumerate(panels):
            ax = axes[i, j]
            for meth in dict.fromkeys(r["method"] for r in sub):
                pts = sorted((r["N"], r.get(m)) for r in sub if r["method"] == meth and r.get(m) is not None)
                if not pts:
                    continue
                x, y = np.array(pts, float).T
                if logy:
                    y = np.where(np.isfinite(y), np.maximum(y, 1e-5), np.nan)
                ax.plot(x, y, label=meth, **style(meth))
            if logy:
                ax.set_yscale("log")
            nmax = max(r["N"] for r in rows)
            if nmax > n_exact:
                ax.axvspan(n_exact + 0.5, nmax + 0.5, color="#eeeeee", zorder=0)
            ax.set_xlim(min(r["N"] for r in rows) - 0.5, nmax + 0.5)
            if i == 0:
                ph = sub[0]["planner_phase"]
                ax.set_title(f"{key}  (planner: {ph})")
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == len(panels) - 1:
                ax.set_xlabel("N   (grey: reference = best known, not certified)")
    hs, ls = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h); ls.append(l)
    fig.legend(hs, ls, loc="lower center", ncol=min(6, len(ls)), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Central planner: QPU (thick) vs classical solvers (thin) vs no planner: the game (dashed); "
                 "classes from easy (left) to hard (right)", y=1.0)
    fig.tight_layout()
    fig.savefig(d / "compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_reference(d: Path, refq: list, classes) -> None:
    """Is the best-known reference plausibly the ground state? For pure SK the mean ground-state reward
    follows R(N) ~ -2 (e_inf + A N^(-2/3)) (1 - eps/2), e_inf = -0.7633 (Parisi)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, eps, J0 in classes:
        pts = sorted((r["N"], r["R_ref_mean"], r["reference"]) for r in refq if r["problem"] == key)
        if not pts:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        line, = ax.plot(x, y, "o-", label=key)
        ex = np.array([p[2] == "exact" for p in pts])
        ax.plot(x[~ex], y[~ex], "o", mfc="white", color=line.get_color())
        if J0 == 0:
            n = np.linspace(max(8, x.min()), x.max(), 200)
            ax.plot(n, -2 * (PARISI_E + FINITE_SIZE_A * n ** (-2 / 3)) * (1 - eps / 2), ":", color=line.get_color(),
                    lw=1, label=f"{key}: Parisi + N^-2/3 (expected)")
    ax.set_xlabel("N")
    ax.set_ylabel("mean reference reward R_ref")
    ax.set_title("reference: filled = exact, open = best known", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "reference_quality.png", dpi=130)
    plt.close(fig)
