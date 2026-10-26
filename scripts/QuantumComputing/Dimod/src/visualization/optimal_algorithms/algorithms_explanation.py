"""How each optimisation algorithm of the benchmark attacks an SK instance, in pictures.

Vocabulary of ../SK_game/README.md. The traces come from traced.py (same update rules as
skgame/optim/solvers.py); figure 11 runs the real benchmark (skgame.optim.benchmark).

  alg01_exact          Gray-code enumeration: one single-flip move per configuration
  alg02_greedy         steepest descent from random starts: basins, and why restarts stop working
  alg03_game           the SK game as an optimiser: freezing on a local minimum
  alg04_sa             simulated annealing: temperature schedule, acceptance, longer anneals
  alg05_pt             parallel tempering: replicas, swaps, random walk in temperature
  alg06_eo             tau-extremal optimisation: flip the most unsatisfied, by rank
  alg07_tabu           tabu search: forced uphill moves, forbidden spins
  alg08_sdp            SDP relaxation + Goemans-Williamson rounding: spins become vectors
  alg09_dwave          the dimod samplers (dwave-samplers): SA, tabu, steepest descent
  alg10_qa             quantum annealing on 8 spins: spectrum, gap, anneal time
  alg11_overview       all solvers on the same instances vs N

    python algorithms_explanation.py            # ~5 min
    python algorithms_explanation.py --steps 10 11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import instance, landscape, plt, save, suptitle  # noqa: E402

import numpy as np  # noqa: E402

import traced as T  # noqa: E402
from skgame.optim.solvers import ExactEnumeration, _greedy  # noqa: E402

N_MID = 24                    # traces: exact optimum known, landscape too large to draw
SEED_MID = 21


def _mid():
    _, P = instance(N_MID, seed=SEED_MID)
    e0 = -ExactEnumeration().solve(P).reward * N_MID / 2       # ground-state energy H_min
    return P.W, e0


def _rand(N, rng):
    return np.where(rng.random(N) < 0.5, 1.0, -1.0)


def alg01_exact():
    _, P8 = instance(8, seed=1)
    g8 = T.gray_code(P8.W, 63)
    _, P16 = instance(16, seed=1)
    g16 = T.gray_code(P16.W)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    S = [-np.ones(8)]
    for k in g8["k"]:
        s = S[-1].copy()
        s[k] = -s[k]
        S.append(s)
    axes[0].imshow(np.array(S).T, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    axes[0].set_xlabel("step (configuration visited)")
    axes[0].set_ylabel("spin i")
    axes[0].set_title("N = 8, first 64 steps: exactly one spin flips per step\n"
                      "(spin k flips when bit k is the lowest set bit of the step)", fontsize=9)
    E = g16["E"] / 16
    axes[1].plot(E, lw=0.2, color="#999999", label="energy per spin of the visited configuration")
    axes[1].plot(np.minimum.accumulate(E), color="k", label="best so far")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("H / N")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"N = 16: all 2^15 = {len(E)} configurations (S_15 fixed by the flip symmetry);\n"
                      "each step costs one O(N) local-field update", fontsize=9)
    suptitle(fig, "Exact enumeration - visits every configuration once: certified ground state, cost N 2^(N-1)")
    return save(fig, HERE, "alg01_exact.png")


def alg02_greedy():
    _, P = instance(14, seed=7)
    L = landscape(P.W)
    Emin = L["E"].min()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for _ in range(25):
        tr = T.steepest_descent(P.W, _rand(14, rng))
        ok = np.isclose(tr["E"][-1], Emin)
        axes[0].plot(tr["E"] / 14, "-o", ms=2, lw=1, color="k" if ok else "#d62728", alpha=0.8 if ok else 0.4)
    axes[0].axhline(Emin / 14, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("single flips (always the most unsatisfied spin)")
    axes[0].set_ylabel("H / N")
    axes[0].set_title("N = 14 (the instance of SK_game step 5), 25 random starts:\n"
                      "black = reached the ground state, red = stuck in another local minimum", fontsize=9)
    Ns, ps = [10, 12, 14, 16, 18, 20, 22, 24], []
    for N in Ns:
        pp = []
        for s in range(6):
            _, Q = instance(N, seed=400 + s)
            e0 = -ExactEnumeration().solve(Q).reward * N / 2
            hit = 0
            for _ in range(300):
                S, _ = _greedy(Q.W, _rand(N, rng))
                hit += np.isclose(-0.5 * S @ Q.W @ S, e0)
            pp.append(hit / 300)
        ps.append(np.mean(pp))
    axes[1].semilogy(Ns, ps, "o-", label="p = P(one descent ends in the ground state)")
    r = np.log(0.01) / np.log1p(-np.array(ps))
    axes[1].semilogy(Ns, 1 / np.array(r), "s--", label="1 / restarts needed for 99 %")
    axes[1].set_xlabel("N")
    axes[1].legend(fontsize=8)
    axes[1].set_title("the basin of the ground state shrinks exponentially:\n"
                      f"restarts needed r = ln 0.01 / ln(1-p): {r[0]:.0f} at N={Ns[0]}, {r[-1]:.0f} at N={Ns[-1]}",
                      fontsize=9)
    suptitle(fig, "Greedy (steepest descent, multistart) - fast, but every run ends in the basin it started in")
    return save(fig, HERE, "alg02_greedy.png")


def alg03_game():
    W, e0 = _mid()
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for _ in range(6):
        tr = T.game_learning(W, 0.05, 400, rng)
        axes[0].plot(tr["E"] / N_MID, lw=1)
        axes[1].plot(tr["flips"], lw=0.8)
    axes[0].axhline(e0 / N_MID, color="k", ls="--", lw=0.8, label="ground state")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("H / N")
    axes[0].legend(fontsize=8)
    axes[0].set_title("S = sign Q;  Q <- (1-alpha) Q + alpha h  (alpha = 0.05):\nthe energy plateaus above the ground state",
                      fontsize=9)
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("spins that changed at step t")
    axes[1].set_yscale("symlog")
    axes[1].set_title("freezing: the flips stop -> a fixed point, i.e. a local minimum", fontsize=9)
    res = {"greedy (1 descent)": [], "game (T=0)": [], "game + annealed noise": [], "game -> greedy polish": []}
    for _ in range(150):
        S, _ = _greedy(W, _rand(N_MID, rng))
        res["greedy (1 descent)"].append(-0.5 * S @ W @ S)
        tr = T.game_learning(W, 0.05, 400, rng)
        res["game (T=0)"].append(tr["E"][-1])
        Sp, _ = _greedy(W, tr["S"].copy())
        res["game -> greedy polish"].append(-0.5 * Sp @ W @ Sp)
        res["game + annealed noise"].append(T.game_learning(W, 0.05, 400, rng, beta0=0.5)["E"][-1])
    for k, v in res.items():
        v = (np.array(v) - e0) / abs(e0)
        axes[2].hist(v * 100, bins=np.linspace(0, 30, 31), histtype="step", lw=1.5,
                     label=f"{k}: P(opt) = {np.mean(v < 1e-9):.2f}")
    axes[2].set_xlabel("residual (H - H_min) / |H_min|  [%]")
    axes[2].legend(fontsize=7)
    axes[2].set_title(f"N = {N_MID}, 150 runs each", fontsize=9)
    suptitle(fig, "The SK game as an optimiser - learning = local search with memory: "
                  "about as good as one greedy descent; polishing its fixed point changes nothing")
    return save(fig, HERE, "alg03_game.png")


def alg04_sa():
    W, e0 = _mid()
    rng = np.random.default_rng(2)
    betas = np.geomspace(0.1, 5.0, 50)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"wspace": 0.45})
    for _ in range(4):
        tr = T.simulated_annealing(W, _rand(N_MID, rng), betas, 4, rng)
        axes[0].plot(np.arange(len(tr["E"])) / 4, tr["E"] / N_MID, lw=0.8)
    axes[0].axhline(e0 / N_MID, color="k", ls="--", lw=0.8)
    ax2 = axes[0].twinx()
    ax2.plot(np.arange(len(betas)), betas, color="#888888", ls=":")
    ax2.set_ylabel("beta = 1/T (dotted)")
    axes[0].set_xlabel("schedule step (4 sweeps each)")
    axes[0].set_ylabel("H / N")
    axes[0].set_title("hot: the energy fluctuates widely (barriers are crossed);\n"
                      "cold: it settles into a minimum", fontsize=9)
    axes[1].semilogx(betas, tr["acceptance"], "o-", ms=3)
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("fraction of flips accepted")
    axes[1].set_title("Metropolis: an unsatisfied spin (dH = 2 lambda < 0) always flips,\n"
                      "a satisfied one with probability exp(-2 beta lambda)", fontsize=9)
    sweeps, ps, pf = [25, 50, 100, 200, 400, 800], [], []
    for sw in sweeps:
        hit = fin = 0
        for _ in range(100):
            b = np.geomspace(0.1, 5.0, 25)
            tr = T.simulated_annealing(W, _rand(N_MID, rng), b, sw // 25, rng)
            hit += tr["best_polished"] <= e0 + 1e-9
            fin += tr["E"][-1] <= e0 + 1e-9
        ps.append(hit / 100)
        pf.append(fin / 100)
    axes[2].plot(sweeps, ps, "o-", label="best state seen + greedy polish (what the solver returns)")
    axes[2].plot(sweeps, pf, "s--", label="last state only (still warm at beta = 5)")
    axes[2].set_xscale("log")
    axes[2].set_xticks(sweeps)
    axes[2].set_xticklabels([str(x) for x in sweeps])
    axes[2].minorticks_off()
    axes[2].set_xlabel("sweeps per run")
    axes[2].set_ylabel("P(ground state), 100 runs")
    axes[2].legend(fontsize=7)
    axes[2].set_title(f"N = {N_MID}: slower cooling -> more often the ground state", fontsize=9)
    suptitle(fig, "Simulated annealing - noise lets the spins climb out of local minima; "
                  "cooling slowly lets them pick the deepest")
    return save(fig, HERE, "alg04_sa.png")


def alg05_pt():
    W, e0 = _mid()
    rng = np.random.default_rng(3)
    betas = np.geomspace(0.2, 5.0, 12)
    tr = T.parallel_tempering(W, betas, 400, rng)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = plt.get_cmap("coolwarm_r")
    for t in range(len(betas)):
        axes[0].plot(tr["E"][:, t] / N_MID, lw=0.6, color=cm(t / (len(betas) - 1)))
    axes[0].plot(tr["best"] / N_MID, color="k", lw=1.5, label="best so far")
    axes[0].axhline(e0 / N_MID, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("sweep")
    axes[0].set_ylabel("H / N at each temperature (red hot, blue cold)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("12 replicas, one per temperature", fontsize=9)
    for r in (0, 5, 11):
        axes[1].plot(tr["temperature_of_replica"][:, r], lw=1, label=f"replica {r}")
    axes[1].set_xlabel("sweep")
    axes[1].set_ylabel("temperature index (0 = hottest)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("each replica random-walks in temperature:\nheated -> leaves its minimum, cooled -> descends into another",
                      fontsize=9)
    axes[2].bar(np.arange(len(betas) - 1), tr["swap_rate"])
    axes[2].set_xlabel("pair (k, k+1) of neighbouring temperatures")
    axes[2].set_ylabel("swap acceptance")
    axes[2].set_title("swap accepted w.p. min(1, exp[(beta_k - beta_k+1)(E_k - E_k+1)]):\n"
                      "a deeper configuration moves to the colder temperature", fontsize=9)
    suptitle(fig, "Parallel tempering - many temperatures at once; low-energy configurations "
                  "flow to the cold end")
    return save(fig, HERE, "alg05_pt.png")


def alg06_eo():
    W, e0 = _mid()
    rng = np.random.default_rng(4)
    tr = T.tau_eo(W, _rand(N_MID, rng), 3000, 1.3, rng)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(tr["E"] / N_MID, lw=0.5, color="#888888", label="current")
    axes[0].plot(np.minimum.accumulate(tr["E"]) / N_MID, color="k", label="best so far")
    axes[0].axhline(e0 / N_MID, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("step (one flip per step, always accepted)")
    axes[0].set_ylabel("H / N")
    axes[0].legend(fontsize=8)
    axes[0].set_title("never settles: it keeps exploring around low energies", fontsize=9)
    k = np.arange(1, N_MID + 1)
    cnt = np.bincount(tr["rank"], minlength=N_MID + 1)[1:]
    axes[1].bar(k, cnt / cnt.sum(), label="ranks chosen")
    axes[1].plot(k, k ** -1.3 / np.sum(k ** -1.3), "k.-", label="k^-tau, tau = 1.3")
    axes[1].set_xlabel("rank k (1 = most unsatisfied spin, smallest lambda)")
    axes[1].set_ylabel("probability")
    axes[1].legend(fontsize=8)
    axes[1].set_title("which spin flips: mostly the worst, sometimes a satisfied one", fontsize=9)
    taus, ps = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0], []
    for tau in taus:
        ps.append(np.mean([T.tau_eo(W, _rand(N_MID, rng), 1500, tau, rng)["best"] <= e0 + 1e-9 for _ in range(30)]))
    axes[2].plot(taus, ps, "o-")
    axes[2].set_xlabel("tau")
    axes[2].set_ylabel("P(ground state), 1500 steps")
    axes[2].set_title("tau too large -> greedy-like, trapped;\ntau too small -> random walk", fontsize=9)
    suptitle(fig, "tau-extremal optimisation - pressure on the least satisfied agents, "
                  "no temperature and no acceptance test")
    return save(fig, HERE, "alg06_eo.png")


def alg07_tabu():
    W, e0 = _mid()
    rng = np.random.default_rng(5)
    S0 = _rand(N_MID, rng)
    tr = T.tabu_search(W, S0, 200, tenure=6)
    sd = T.steepest_descent(W, S0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(tr["E"] / N_MID, lw=1, label="tabu (tenure 6)")
    axes[0].plot(sd["E"] / N_MID, lw=2, label="steepest descent from the same start (stops)")
    axes[0].axhline(e0 / N_MID, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("H / N")
    axes[0].legend(fontsize=8)
    axes[0].set_title("at a local minimum tabu still moves:\nit flips the least-bad spin, uphill if needed", fontsize=9)
    M = np.zeros((N_MID, 80))
    for t, k in enumerate(tr["moved"][:80]):
        M[k, t:t + 7] = np.linspace(1, 0.2, 7)[: min(7, 80 - t)]
    axes[1].imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("spin")
    axes[1].set_title("tabu list: a flipped spin is frozen for 6 steps (dark -> light),\n"
                      "so the search cannot undo it and fall back", fontsize=9)
    ten, ps = [0, 1, 2, 4, 6, 8, 12], []
    for tn in ten:
        ps.append(np.mean([T.tabu_search(W, _rand(N_MID, rng), 300, tn)["best"] <= e0 + 1e-9 for _ in range(30)]))
    axes[2].plot(ten, ps, "o-")
    axes[2].set_xlabel("tenure")
    axes[2].set_ylabel("P(ground state), 300 steps")
    axes[2].set_title("tenure 0 = steepest descent that oscillates;\nsome memory is what makes it work", fontsize=9)
    suptitle(fig, "Tabu search - deterministic escape from local minima by short-term memory of moves")
    return save(fig, HERE, "alg07_tabu.png")


def alg08_sdp():
    W, e0 = _mid()
    rng = np.random.default_rng(6)
    k = int(np.ceil(np.sqrt(2 * N_MID))) + 1
    sd = T.sdp_mixing(W, k, 80, rng)
    S, E = T.gw_rounding(W, sd["V"], 400, rng)
    Ep = np.array([-0.5 * s @ W @ s for s in (_greedy(W, x.copy())[0] for x in S)])
    sd3 = T.sdp_mixing(W, 3, 80, rng)
    fig = plt.figure(figsize=(15, 4.4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(sd["relaxed_energy"] / N_MID, "o-", ms=2)
    ax0.axhline(e0 / N_MID, color="k", ls="--", lw=0.8, label="ground state H_min/N")
    ax0.set_xlabel("mixing-method sweep")
    ax0.set_ylabel("relaxed energy / N")
    ax0.legend(fontsize=8)
    ax0.set_title(f"spins -> unit vectors in R^{k}: v_i <- normalise(sum_j W_ij v_j)\n"
                  "the relaxed minimum is BELOW H_min: a certified lower bound", fontsize=9)
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    V = sd3["V"]
    r = rng.standard_normal(3)
    s = np.sign(V @ r)
    ax1.scatter(*V.T, c=np.where(s > 0, "#d62728", "#1f77b4"), s=30)
    for v in V:
        ax1.plot([0, v[0]], [0, v[1]], [0, v[2]], color="#bbbbbb", lw=0.5)
    ax1.set_title("rank-3 picture: each agent is a direction;\nrandom hyperplane r: S_i = sign(v_i . r)", fontsize=9)
    ax2 = fig.add_subplot(1, 3, 3)
    bins = np.linspace(e0 / N_MID - 0.02, E.max() / N_MID, 40)
    ax2.hist(E / N_MID, bins=bins, histtype="step", lw=1.5, label="after rounding")
    ax2.hist(Ep / N_MID, bins=bins, histtype="step", lw=1.5, label="rounding + greedy polish")
    ax2.axvline(e0 / N_MID, color="k", ls="--", lw=0.8, label="ground state")
    ax2.axvline(sd["relaxed_energy"][-1] / N_MID, color="#9467bd", lw=1.2, label="SDP bound")
    ax2.set_xlabel("H / N")
    ax2.legend(fontsize=8)
    ax2.set_title(f"400 hyperplanes: P(ground) = {np.mean(E <= e0 + 1e-9):.2f} raw, "
                  f"{np.mean(Ep <= e0 + 1e-9):.2f} polished", fontsize=9)
    suptitle(fig, "SDP relaxation + Goemans-Williamson - solve a smooth problem exactly, "
                  "then cut it back to spins")
    return save(fig, HERE, "alg08_sdp.png")


def alg09_dwave():
    from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver, TabuSampler
    _, P = instance(64, seed=31)
    bqm = P.to_bqm()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    sa = SimulatedAnnealingSampler()
    allE = {}
    for sw in (10, 100, 1000):
        ss = sa.sample(bqm, num_reads=200, num_sweeps=sw, seed=1)
        allE[f"SA {sw} sweeps"] = ss.record.energy
        beta_range = ss.info.get("beta_range")
    ts = TabuSampler()
    for tm in (2, 20):
        allE[f"tabu {tm} ms"] = ts.sample(bqm, num_reads=50, timeout=tm, seed=1).record.energy
    allE["steepest descent"] = SteepestDescentSolver().sample(bqm, num_reads=200, seed=1).record.energy
    ref = min(e.min() for e in allE.values())
    for name, Es in allE.items():
        ax = axes[0] if name.startswith("SA") else axes[1]
        ax.hist((Es - ref) / abs(ref) * 100, bins=np.linspace(0, 25, 50), histtype="step", lw=1.5,
                label=f"{name}: P(best) = {np.mean(Es <= ref + 1e-9):.2f}")
    for ax, t in zip(axes[:2], ("dwave SimulatedAnnealingSampler", "TabuSampler vs SteepestDescentSolver")):
        ax.set_xlabel("residual vs best found [%]")
        ax.legend(fontsize=7)
        ax.set_title(t, fontsize=9)
    axes[2].axis("off")
    axes[2].text(0, 1, "\n".join([
        f"N = 64, same BQM for all (energy H in Ising form).",
        "",
        "SimulatedAnnealingSampler (C++):",
        "  sweeps visit the spins in order, Metropolis test",
        f"  beta schedule geometric over beta_range",
        f"  = {np.round(beta_range, 3) if beta_range is not None else 'auto'} (set from the",
        "  smallest and largest possible dH of the instance)",
        "  hot enough to flip anything, cold enough to freeze",
        "",
        "TabuSampler: MST2 multistart tabu search",
        "  tenure = min(20, N/4), restarts until",
        "  `timeout` ms per read",
        "",
        "SteepestDescentSolver: greedy descent,",
        "  identical to alg02, ends in a local minimum",
    ]), va="top", family="monospace", fontsize=9)
    suptitle(fig, "dimod samplers (dwave-samplers) - the same algorithms behind the dimod interface "
                  "the QPU uses")
    return save(fig, HERE, "alg09_dwave.png")


def alg10_qa():
    _, P = instance(8, seed=1)
    W = P.W
    Ez, S = T.diag_energies(W)
    e0 = Ez.min()
    sg = np.linspace(0, 1, 101)
    spec = T.qa_spectrum(W, sg, 6)
    gap = spec[:, 2] - spec[:, 0]                      # ground pair is degenerate at s=1 (S, -S)
    Ts = [0.5, 1, 2, 5, 10, 20, 50, 100, 300, 1000]
    pg = [T.qa_evolve(W, t, n_steps=600)[np.isclose(Ez, e0)].sum() for t in Ts]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for l in range(spec.shape[1]):
        axes[0].plot(sg, spec[:, l], lw=1)
    j = np.argmin(gap[5:-5]) + 5
    axes[0].annotate(f"min gap {gap[j]:.3f}", (sg[j], spec[j, 0]), xytext=(0.1, spec[:, 0].min()),
                     arrowprops=dict(arrowstyle="->"))
    axes[0].set_xlabel("s = t / T")
    axes[0].set_ylabel("energy levels of H(s)")
    axes[0].set_title("H(s) = -(1-s) sum_i sigma^x_i + s H_P / max|W|\n"
                      "s=0: all configurations superposed; s=1: classical energies", fontsize=9)
    axes[1].semilogx(Ts, pg, "o-")
    axes[1].set_xlabel("anneal time T (units of hbar / max|W|)")
    axes[1].set_ylabel("P(ground state) at the end")
    axes[1].set_title("slow enough (T >> 1/gap^2) -> adiabatic: the system stays\n"
                      "in the instantaneous ground state and ends in the optimum", fontsize=9)
    for t, c in ((1, "#d62728"), (10, "#ff7f0e"), (1000, "#2ca02c")):
        p = T.qa_evolve(W, t, n_steps=600)
        order = np.argsort(Ez)
        axes[2].plot((Ez[order] - e0) / abs(e0), np.cumsum(p[order]), color=c, label=f"T = {t}")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_xlabel("residual (H - H_min) / |H_min|")
    axes[2].set_ylabel("P(read has residual <= x)")
    axes[2].legend(fontsize=8)
    axes[2].set_title("each read is one configuration drawn from |psi(T)|^2;\n"
                      "fast anneals spread the reads over many local minima", fontsize=9)
    suptitle(fig, "Quantum annealing (exact simulation, N = 8, idealised linear schedule) - "
                  "tunnelling instead of climbing")
    return save(fig, HERE, "alg10_qa.png")


def alg11_overview():
    from skgame.optim.benchmark import BenchmarkConfig, run_benchmark
    from skgame.optim.solvers import default_solvers
    cfg = BenchmarkConfig(Ns=(12, 16, 20, 24, 32, 48, 64), n_instances=6, n_runs=4, n_exact=24, ref_sweeps=3000)
    sv = default_solvers("quick")
    res = run_benchmark(sv, cfg, log=lambda *a: None)
    m = res.metrics()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for s, name in enumerate(res.solvers):
        if name == "exact":
            continue
        axes[0].plot(res.Ns, m["p_median"][s], "o-", ms=3, label=name)
        axes[1].semilogy(res.Ns, np.maximum(m["residual_median"][s], 1e-5), "o-", ms=3, label=name)
    axes[0].axvline(24.5, color="#888888", ls=":", lw=1)
    axes[0].text(25, 0.05, "exact reference | best known", fontsize=8, color="#555555")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("median success probability per run")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("median residual (floored at 1e-5)")
    suptitle(fig, "All solvers on the same SK instances (quick budgets, 6 instances x 4 runs) - "
                  "skgame.optim.benchmark")
    return save(fig, HERE, "alg11_overview.png")


STEPS = {1: alg01_exact, 2: alg02_greedy, 3: alg03_game, 4: alg04_sa, 5: alg05_pt, 6: alg06_eo,
         7: alg07_tabu, 8: alg08_sdp, 9: alg09_dwave, 10: alg10_qa, 11: alg11_overview}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=sorted(STEPS))
    a = ap.parse_args(argv)
    for s in a.steps:
        print("wrote", STEPS[s]())


if __name__ == "__main__":
    main()
