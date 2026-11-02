"""Choose the chain-strength prefactor p for every N by testing it on the QPU.

    python main_simulation.py coupling-scan --out results/full2 --Ns 8:48:8 104 --check
    python main_simulation.py coupling-scan --out results/full2 --Ns 8:48:8 104 --budget-seconds 7

For every back-end and every N, the same instance(s) are annealed with each prefactor p of a grid.
For each (back-end, N, p) the scan records how many chains broke in every read.

Rule (find_best_coupling_per_problem / choose_coupling):
  * among the p whose mean fraction of broken chains is <= threshold (default 0: no chain broken
    in any read), take the SMALLEST: larger p would also keep chains intact but squeezes the SK
    couplings further into the QPU's noise (auto_scale divides everything by ~c);
  * if no p reaches the threshold, take the p with the fewest broken chains (ties: smaller p).

The chosen table {back-end: {N: p}} is written to <out>/coupling_scan/best_coupling.json and into
config.json (chain_strength_by_N), so `qpu` then uses, call by call, the p of its back-end and N:
values between scanned N are interpolated linearly, values beyond the largest scanned N keep the
last one. Curves: <out>/coupling_scan/coupling_scan.png (broken chains vs p per N, chosen p
marked; chosen p vs N), table: coupling_scan.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_PREFACTORS = (0.75, 1.0, 1.414, 2.0, 3.0, 4.0, 6.0)


def scan_path(out: Path, label: str, key: str, N: int, p: float, b: int) -> Path:
    return out / "coupling_scan" / label / key / f"N{N:04d}_p{p:g}_i{b}.json"


def choose_coupling(rows: list[dict], threshold: float = 0.0) -> tuple[float, str]:
    """rows: [{"p", "broken_fraction"}...] for ONE back-end and N. Returns (p, reason)."""
    ok = [r for r in rows if r["broken_fraction"] <= threshold + 1e-15]
    if ok:
        best = min(ok, key=lambda r: r["p"])
        why = f"smallest p with broken-chain fraction <= {threshold:g}"
        if best["p"] == min(r["p"] for r in rows):
            why += " (already at the SMALLEST p tested: a smaller p may also work)"
        return best["p"], why
    best = min(rows, key=lambda r: (r["broken_fraction"], r["p"]))
    why = "no p reached the threshold: fewest broken chains"
    if best["p"] == max(r["p"] for r in rows):
        why += " (at the LARGEST p tested: extend the grid upwards)"
    return best["p"], why


def coupling_for(table: dict, label: str, N: int, default: float) -> float:
    """Prefactor for back-end `label` at size N from a {label: {N: p}} table: exact N, else linear
    interpolation between the nearest scanned N, else the nearest end (constant extrapolation)."""
    t = table.get(label) or {}
    if not t:
        return default
    xs = sorted(int(k) for k in t)
    ys = [float(t[str(x)] if str(x) in t else t[x]) for x in xs]
    return float(np.interp(N, xs, ys))


def find_best_coupling_per_problem(out: Path, cfg, Ns, prefactors=DEFAULT_PREFACTORS, specs=None, keys=None,
                                   n_instances: int = 1, num_reads: int = 200, threshold: float = 0.0,
                                   budget=None, check_only: bool = False, apply: bool = True, log=print) -> dict:
    import main_simulation as ms
    from skgame.optim.problem import PlannerProblem

    specs = specs or [b for b in cfg.backends if b.startswith("qpu:")]
    classes = {k: (k, e, j) for k, e, j in ms._classes(cfg)}
    keys = keys or [next((k for k in classes if k == "eps0_J00"), next(iter(classes)))]
    budget = budget or ms.QPUBudget(None)
    backends = []
    for spec in specs:
        be = ms.make_backend(out, cfg, spec)
        be.settings.num_reads = num_reads
        backends.append(be)
    # order: N outer, so a budget stop leaves complete Ns behind
    todo = []
    for N in sorted(set(Ns)):
        for be in backends:
            limit = be.largest_problem()
            if limit is not None and N > limit:
                log(f"[coupling] {be.spec}: N={N} above the largest clique ({limit}), skipped")
                continue
            for key in keys:
                for p in prefactors:
                    for b in range(n_instances):
                        if not scan_path(out, be.label, key, N, p, b).exists():
                            todo.append((be, key, N, p, b))
    log(f"[coupling] {len(todo)} calls to do: Ns={sorted(set(Ns))}, p={list(prefactors)}, "
        f"{n_instances} instance(s) of {keys}, {num_reads} reads, back-ends {specs}")
    if check_only:
        total = 0.0
        for be, key, N, p, b in todo:
            total += ms._call_estimate(be, N, cfg)
        log(f"[coupling] estimated QPU access time: {total:.2f} s (spends nothing)")
        return {}
    for n, (be, key, N, p, b) in enumerate(todo):
        est = ms._call_estimate(be, N, cfg)
        if not budget.allows(est):
            log(f"[coupling] budget: spent {budget.spent:.3f}s of {budget.limit:.3f}s; stopping before "
                f"{be.label} N={N} p={p:g} (rerun to continue)")
            break
        _, eps, J0 = classes[key]
        P = PlannerProblem(ms.ensure_instances(out, cfg, key, eps, J0, N)[b])
        be.settings.chain_strength_prefactor = float(p)
        res = be.sample(P, rng=np.random.SeedSequence([cfg.seed, 7, N, b]),
                        label=f"skgame coupling-scan {be.label} {key} N={N} p={p:g} i={b}")
        res.update(key=key, eps=eps, J0=J0, N=N, instance=b, backend=be.spec, prefactor=float(p))
        res["info"]["estimated_access_s"] = est
        phys = res.pop("_physical", None)
        path = scan_path(out, be.label, key, N, p, b)
        if phys is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path.with_name(path.stem + "_physical.npz"), **phys)
            res["info"]["physical_problem_file"] = path.stem + "_physical.npz"
        ms._write_json(path, res)
        acc = res["info"].get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
        budget.charge(acc if acc > 0 else est, est)
        log(f"[coupling] {n + 1:3d}/{len(todo)} {be.label:20s} N={N:4d} p={p:<6g} "
            f"broken/read={_mean_broken(res):6.3f} qpu_access={acc * 1e3:.1f}ms (total {budget.spent:.3f}s)")
    return summarize_coupling_scan(out, cfg, threshold=threshold, apply=apply, log=log)


def _mean_broken(res) -> float:
    h = res["info"].get("broken_chains_hist") or {}
    return sum(int(k) * v for k, v in h.items()) / max(1, sum(h.values()))


def summarize_coupling_scan(out: Path, cfg, threshold: float = 0.0, apply: bool = True, log=print) -> dict:
    """Read every scan result, choose p per (back-end, N), write table/curves, store in config."""
    import main_simulation as ms
    root = out / "coupling_scan"
    acc = {}
    for f in sorted(root.glob("*/*/N*_p*_i*.json")) if root.exists() else []:
        r = json.loads(f.read_text())
        a = acc.setdefault((f.parent.parent.name, r["N"], r["prefactor"]),
                           {"reads": 0, "broken": 0, "zero": 0, "best": -np.inf, "chain_strength": [],
                            "c_over_max": [], "prog_rms": []})
        info = r["info"]
        if info.get("chain_strength") and info.get("max_abs_logical_coupling"):
            a["c_over_max"].append(info["chain_strength"] / info["max_abs_logical_coupling"])
        pf = f.with_name(f.stem + "_physical.npz")
        if pf.exists() and info.get("embedding_used"):
            ph = np.load(pf)
            own = {q: int(i) for i, ch in info["embedding_used"].items() for q in ch}
            logi = np.array([v for x, y, v in zip(ph["u"], ph["v"], ph["J"]) if own[int(x)] != own[int(y)]])
            div = info["programmed"]["auto_scale_divisor"]
            if logi.size and div:
                a["prog_rms"].append(float(np.sqrt(np.mean(logi ** 2)) / div))
        h = r["info"].get("broken_chains_hist") or {}
        a["reads"] += sum(h.values())
        a["broken"] += sum(int(k) * v for k, v in h.items())
        a["zero"] += h.get("0", 0)
        a["best"] = max(a["best"], r.get("polished_best_reward", r["best_reward"]))
        a["chain_strength"].append(r["info"].get("chain_strength"))
    rows = []
    for (lab, N, p), a in sorted(acc.items()):
        rows.append({"backend": lab, "N": N, "p": p, "reads": a["reads"],
                     "broken_per_read": a["broken"] / max(1, a["reads"]),
                     "broken_fraction": a["broken"] / max(1, a["reads"]) / N,
                     "frac_reads_no_break": a["zero"] / max(1, a["reads"]),
                     "best_reward_polished": a["best"],
                     "chain_strength": float(np.mean([c for c in a["chain_strength"] if c is not None] or [np.nan])),
                     "c_over_max_coupling": float(np.mean(a["c_over_max"])) if a["c_over_max"] else None,
                     "programmed_logical_rms": float(np.mean(a["prog_rms"])) if a["prog_rms"] else None})
    best = {}
    for lab in dict.fromkeys(r["backend"] for r in rows):
        for N in sorted({r["N"] for r in rows if r["backend"] == lab}):
            sel = [r for r in rows if r["backend"] == lab and r["N"] == N]
            p, why = choose_coupling(sel, threshold)
            best.setdefault(lab, {})[str(N)] = p
            for r in sel:
                r["chosen"] = r["p"] == p
                r["reason"] = why if r["p"] == p else ""
    if not rows:
        log("[coupling] no scan results yet")
        return {}
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "coupling_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    payload = {"rule": f"smallest p with broken-chain fraction <= {threshold:g}, else fewest broken chains",
               "threshold": threshold, "best": best,
               "use": "N between scanned values: linear interpolation; beyond: nearest scanned value"}
    ms._write_json(root / "best_coupling.json", payload)
    _plot(root, rows, best)
    for lab, t in best.items():
        log(f"[coupling] best prefactor for {lab}: " + ", ".join(f"N={k}: p={v:g}" for k, v in t.items()))
        for r in rows:
            if r["backend"] == lab and r.get("chosen") and "(" in r["reason"]:
                log(f"[coupling]   N={r['N']}: {r['reason']}")
    if apply:
        cfg.chain_strength_by_N = best
        ms.save_config(out, cfg)
        log(f"[coupling] stored in {out / 'config.json'} (chain_strength_by_N): `qpu` now uses these values")
    log(f"[coupling] -> {root / 'coupling_scan.csv'}, {root / 'coupling_scan.png'}, {root / 'best_coupling.json'}")
    return best


def _plot(root: Path, rows: list[dict], best: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(dict.fromkeys(r["backend"] for r in rows))
    fig, axes = plt.subplots(4, len(labs), figsize=(5.2 * len(labs), 13), squeeze=False)
    cm = plt.get_cmap("viridis")
    Ns_all = sorted({r["N"] for r in rows})
    for j, lab in enumerate(labs):
        for N in sorted({r["N"] for r in rows if r["backend"] == lab}):
            sel = sorted((r for r in rows if r["backend"] == lab and r["N"] == N), key=lambda r: r["p"])
            x = [r["p"] for r in sel]
            col = cm(Ns_all.index(N) / max(1, len(Ns_all) - 1))
            axes[0, j].plot(x, [r["broken_fraction"] for r in sel], "o-", color=col, label=f"N={N}")
            axes[1, j].plot(x, [r["frac_reads_no_break"] for r in sel], "o-", color=col, label=f"N={N}")
            if all(r.get("programmed_logical_rms") for r in sel):
                axes[3, j].plot(x, [r["programmed_logical_rms"] for r in sel], "o-", color=col, label=f"N={N}")
            ch = [r for r in sel if r.get("chosen")]
            if ch:
                axes[0, j].plot(ch[0]["p"], ch[0]["broken_fraction"], "*", ms=14, color=col, mec="k")
                axes[1, j].plot(ch[0]["p"], ch[0]["frac_reads_no_break"], "*", ms=14, color=col, mec="k")
        axes[0, j].set_title(lab)
        axes[0, j].set_ylabel("fraction of chains broken (per read)")
        axes[0, j].set_yscale("symlog", linthresh=1e-3)
        axes[1, j].set_ylabel("fraction of reads with no broken chain")
        for i in (0, 1):
            axes[i, j].set_xlabel("chain-strength prefactor p (star: chosen)")
        axes[0, j].legend(fontsize=6, ncol=2)
        axes[3, j].set_xlabel("chain-strength prefactor p")
        axes[3, j].set_ylabel("rms of programmed SK couplings\n(QPU range [-1, 1])")
        axes[3, j].set_yscale("log")
        axes[3, j].axhspan(0.01, 0.03, color="#dddddd", zorder=0)
        axes[3, j].set_title("larger p: chains hold, but the SK problem shrinks towards the\n"
                             "control-error band (grey, order 1e-2)", fontsize=9)
        t = best.get(lab, {})
        xs = sorted(int(k) for k in t)
        axes[2, j].plot(xs, [t[str(x)] for x in xs], "s-", color="k")
        axes[2, j].set_xlabel("N (number of agents)")
        axes[2, j].set_ylabel("chosen prefactor p")
        axes[2, j].set_title("used by the main run (interpolated between scanned N)", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "coupling_scan.png", dpi=130)
    plt.close(fig)
