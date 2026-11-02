"""Flux-bias shimming of the QPU for the embedded SK problem, and a record of the machine's state.

Reproduces, for this project, the calibration-refinement procedure of D-Wave's shimming tutorial
(Chern et al., "Tutorial: calibration refinement in quantum annealing", Front. Comput. Sci. 2023,
github.com/dwavesystems/shimming-tutorial, example1_1) and the superspin-level variant of
Teza, Campaioli, Avesani, Raz, Nat. Commun. (2026), Methods, Eq. (8).

Why it applies to SK. The problem has no field (h = 0), so H(S) = H(-S): the exact distribution
of every read is symmetric under the global flip, and every physical qubit must have <s_q> = 0.
A measured magnetisation m_q != 0 is therefore a device bias (flux offsets, crosstalk, leakage
from the strong chain couplers). The shim is the iterative feedback
        Phi_q  <-  Phi_q - alpha_Phi * m_q                       (level "qubit", tutorial)
        Phi_q  <-  Phi_q - alpha_Phi * m_{i(q)},  q in chain i   (level "chain", Teza et al.)
with m_i the mean magnetisation of chain i's qubits; flux biases are passed with
`flux_biases` (units of Phi0) and `flux_drift_compensation=True`. Optional random spin-reversal
gauges (S_i -> g_i S_i, J_ij -> g_i g_j J_ij) change the problem every iteration so that the
feedback learns the device, not the instance. Coupler shimming (tutorial example1_2) needs groups of
symmetry-equivalent couplers ("orbits"); random SK couplings have none, so it is not applied.

    python main_simulation.py shim --out results/full2 --backend qpu:zephyr:clique --N 64 --check
    python main_simulation.py shim --out results/full2 --backend qpu:zephyr:clique --N 64 \\
           --iterations 30 --budget-seconds 1.5
    python main_simulation.py shim --out results/full2 --backend qpu:zephyr:clique --N 64 \\
           --iterations 3 --alpha-phi 0       # diagnosis only: measures the biases, changes nothing

Outputs in <out>/shims/<backend>/: fbos.json (qubit -> offset, applied to later qpu calls of that
back-end when --apply), history.json and shim_history.png (per iteration: rms and max |m_q|,
rms offset, chain-break fraction, best energy, QPU time).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def shim_dir(out: Path, label: str) -> Path:
    return out / "shims" / label


def load_fbos(out: Path, label: str) -> dict:
    p = shim_dir(out, label) / "fbos.json"
    return {int(k): float(v) for k, v in json.loads(p.read_text())["fbos"].items()} if p.exists() else {}


def run_shimming(out: Path, cfg, spec: str, N: int, iterations: int = 30, alpha_phi: float = 1e-5,
                 num_reads: int = 100, annealing_time: float | None = None, level: str = "qubit",
                 gauge: bool = True, key: str | None = None, budget=None, check_only: bool = False,
                 apply: bool = True, seed: int = 11, log=print) -> dict:
    import main_simulation as ms
    from dwave_utils.coupling_scan import prefactor_for
    from skgame.optim.problem import PlannerProblem

    be = ms.make_backend(out, cfg, spec)
    if be.device != "qpu" and not spec.startswith("sim:"):
        raise ValueError("shimming needs an embedded back-end")
    be.settings.num_reads = num_reads
    if annealing_time is not None:
        be.settings.annealing_time = annealing_time
    be.settings.chain_strength_prefactor = prefactor_for(cfg, be.label, key or "eps0_J00", N)
    be.record_qubit_magnetization = True
    classes = {k: (k, e, j) for k, e, j in ms._classes(cfg)}
    key = key or ("eps0_J00" if "eps0_J00" in classes else next(iter(classes)))
    _, eps, J0 = classes[key]
    J = ms.ensure_instances(out, cfg, key, eps, J0, N)[0]
    emb = be.embedding(N)
    budget = budget or ms.QPUBudget(None)
    est = ms._call_estimate(be, N, cfg)
    log(f"[shim] {spec} N={N}: {iterations} iterations x {num_reads} reads, alpha_Phi={alpha_phi:g}, "
        f"level={level}, gauge={gauge}, p={be.settings.chain_strength_prefactor:g}; "
        f"~{est * 1e3:.0f} ms per iteration, ~{iterations * est:.2f} s in total")
    if check_only:
        return {}
    d = shim_dir(out, be.label)
    d.mkdir(parents=True, exist_ok=True)
    fbos = load_fbos(out, be.label)                       # continue from previous shimming
    hist_path = d / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    rng = np.random.default_rng([seed, N, len(history)])
    for it in range(iterations):
        if not budget.allows(est):
            log(f"[shim] budget: spent {budget.spent:.3f}s of {budget.limit:.3f}s; stopping (rerun to continue)")
            break
        g = np.where(rng.random(N) < 0.5, 1.0, -1.0) if gauge else np.ones(N)
        Pg = PlannerProblem(J * np.outer(g, g))           # gauge-transformed instance, same physics
        be.flux_biases = dict(fbos)
        be.flux_biases_source = str(d / "fbos.json")
        res = be.sample(Pg, rng=rng.integers(2**31), label=f"skgame shim {be.label} N={N} it={len(history)}")
        m = {int(q): v for q, v in res["info"]["qubit_magnetization"].items()}
        if level == "chain":
            for i, chain in emb.items():
                mi = float(np.mean([m[q] for q in chain]))
                for q in chain:
                    fbos[q] = fbos.get(q, 0.0) - alpha_phi * mi
        else:
            for q, mq in m.items():
                fbos[q] = fbos.get(q, 0.0) - alpha_phi * mq
        mv = np.array(list(m.values()))
        mc = np.array([np.mean([m[q] for q in ch]) for ch in emb.values()])
        acc = res["info"].get("timing_us", {}).get("qpu_access_time", 0.0) / 1e6
        budget.charge(acc if acc > 0 else est, est)
        fv = np.array(list(fbos.values()))
        Sb = np.array([1.0 if ch == "+" else -1.0 for ch in res["best_S"]]) * g if res.get("best_S") else None
        row = {"iteration": len(history), "rms_m_qubit": float(np.sqrt(np.mean(mv ** 2))),
               "max_abs_m_qubit": float(np.abs(mv).max()), "rms_m_chain": float(np.sqrt(np.mean(mc ** 2))),
               "rms_fbo": float(np.sqrt(np.mean(fv ** 2))), "max_abs_fbo": float(np.abs(fv).max()),
               "chain_break_fraction": res["info"].get("chain_break_fraction_mean"),
               "best_energy": float(PlannerProblem(J).energy(Sb)) if Sb is not None else None,
               "qpu_access_s": acc, "num_reads": num_reads, "alpha_phi": alpha_phi, "level": level, "gauge": bool(gauge),
               "graph_sha1": res["info"]["solver_properties"].get("graph_sha1"),
               "problem_id": res["info"].get("problem_id")}
        history.append(row)
        log(f"[shim] it {row['iteration']:3d}: rms m_q={row['rms_m_qubit']:.4f} max|m_q|={row['max_abs_m_qubit']:.3f} "
            f"rms m_chain={row['rms_m_chain']:.4f} rms Phi={row['rms_fbo']:.2e} breaks={row['chain_break_fraction']} "
            f"qpu={acc * 1e3:.1f}ms (total {budget.spent:.3f}s)")
        ms._write_json(d / "fbos.json", {"backend": be.spec, "solver": be.solver_id(), "N": N, "level": level,
                                         "alpha_phi": alpha_phi, "iterations": len(history),
                                         "graph_sha1": row["graph_sha1"], "fbos": {str(q): v for q, v in fbos.items()}})
        ms._write_json(hist_path, history)
    if history:
        _plot(d, history)
        log(f"[shim] -> {d / 'fbos.json'}, {hist_path}, {d / 'shim_history.png'}")
    if apply and fbos:
        files = dict(ms.load_config(out).flux_biases_files or {})
        files[be.label] = str((d / "fbos.json").relative_to(out))
        ms.update_config(out, cfg, flux_biases_files=files)
        log(f"[shim] stored in config.json: `qpu` calls of {be.label} now use these flux-bias offsets")
    return {"fbos": fbos, "history": history}


def _plot(d: Path, history: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [h["iteration"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(it, [h["rms_m_qubit"] for h in history], "o-", ms=3, label="rms <s_q> (qubits)")
    axes[0].plot(it, [h["rms_m_chain"] for h in history], "s-", ms=3, label="rms chain magnetisation")
    axes[0].plot(it, [h["max_abs_m_qubit"] for h in history], ":", label="max |<s_q>|")
    R = history[-1].get("num_reads")
    if R:      # with h = 0 each read gives s_q = +-1 at random: |m_q| below ~1/sqrt(R) is shot noise, not bias
        axes[0].axhline(1 / np.sqrt(R), color="grey", ls="--", lw=0.8, label="shot-noise floor 1/sqrt(reads)")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("magnetisation (target 0)")
    axes[0].legend(fontsize=8)
    axes[1].plot(it, [h["rms_fbo"] for h in history], "o-", ms=3, label="rms offset")
    axes[1].plot(it, [h["max_abs_fbo"] for h in history], ":", label="max |offset|")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("flux-bias offset [Phi0]")
    axes[1].legend(fontsize=8)
    axes[2].plot(it, [h["chain_break_fraction"] or 0 for h in history], "o-", ms=3, label="chain-break fraction")
    ax2 = axes[2].twinx()
    ax2.plot(it, [h["best_energy"] for h in history], "k.", label="best energy")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("chain-break fraction")
    ax2.set_ylabel("best energy H")
    fig.suptitle("Flux-bias shimming: device biases measured on the Z2-symmetric SK problem (target <s_q> = 0)")
    fig.tight_layout()
    fig.savefig(d / "shim_history.png", dpi=130)
    plt.close(fig)
