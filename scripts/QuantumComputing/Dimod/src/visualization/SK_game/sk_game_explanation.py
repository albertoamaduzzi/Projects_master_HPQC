"""Step-by-step pictures of the SK game and of its optimisation problem.

Every figure is computed from the model code (skgame) on real draws of the interaction ensemble.
The vocabulary is defined in README.md (Glossary) and in ../common.py; the algorithm pages in
../optimal_algorithms use exactly the same terms.

  step01_couplings          J_ij: symmetric, independent, antisymmetric (eps = 0, 1, 2)
  step02_network            the agents as a network; frustrated triangles
  step03_configuration      a configuration S: local fields h_i and satisfactions lambda_i
  step04_single_flip        a single-flip move changes the energy by 2 lambda_k
  step05_landscape          every one of the 2^N configurations: energy vs distance to the ground state
  step06_basins             local minima and the size of their basins
  step07_metastable_count   number of local minima grows like e^{0.199 N}
  step08_ground_state       ground-state energy per spin -> Parisi value -0.7632
  step09_game_dynamics      the learning dynamics in each phase (fixed points, cycles, chaos, paramagnet)
  step10_satisficing        the game's fixed points are local minima, not the ground state

    python sk_game_explanation.py             # all figures into figures/ (~1 min)
    python sk_game_explanation.py --steps 5 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (DOWN, NEG, POS, UP, all_states, energy, hamming_to, instance, landscape,  # noqa: E402
                    plt, satisfaction, save, suptitle)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from skgame.model import GameParams, SKGame, WindowRecorder  # noqa: E402
from skgame.optim.regimes import regimes_by_name  # noqa: E402
from skgame.optim.solvers import ExactEnumeration, ParallelTempering, _greedy  # noqa: E402
from skgame.theory.analytic import PARISI_E0  # noqa: E402


def step01_couplings(N: int = 16):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for c, eps in enumerate((0.0, 1.0, 2.0)):
        J, _ = instance(N, eps, seed=1)
        v = np.abs(J).max()
        im = axes[0, c].imshow(J, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[0, c].set_title(f"eps = {eps:g}: J_ij (row i receives from j)")
        axes[0, c].set_xticks([])
        axes[0, c].set_yticks([])
        Jb, _ = instance(400, eps, seed=2)
        iu = np.triu_indices(400, 1)
        x, y = Jb[iu] * np.sqrt(400), Jb.T[iu] * np.sqrt(400)
        axes[1, c].plot(x[::40], y[::40], ".", ms=2, alpha=0.5)
        eta = np.corrcoef(x, y)[0, 1]
        axes[1, c].set_title(f"J_ji vs J_ij (x sqrt N): correlation eta = {eta:+.2f}")
        axes[1, c].set_xlabel("sqrt(N) J_ij")
        axes[1, c].set_aspect("equal")
    axes[1, 0].set_ylabel("sqrt(N) J_ji")
    fig.colorbar(im, ax=axes[0, :], fraction=0.02)
    suptitle(fig, "Step 1 - couplings: J = (1-eps/2) J^S + (eps/2) J^A.  eps=0 reciprocal (eta=1), "
                  "eps=1 independent (eta=0), eps=2 antisymmetric (eta=-1)")
    return save(fig, HERE, "step01_couplings.png")


def _frustrated_triangle(W):
    N = W.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                if W[i, j] * W[j, k] * W[k, i] < 0:
                    return i, j, k
    return None


def _draw_net(ax, W, S=None, highlight=None, title=""):
    N = W.shape[0]
    G = nx.complete_graph(N)
    pos = nx.circular_layout(G)
    m = np.abs(W).max()
    for i, j in G.edges():
        ax.plot(*zip(pos[i], pos[j]), color=POS if W[i, j] > 0 else NEG, lw=0.3 + 3.5 * abs(W[i, j]) / m,
                alpha=0.75, zorder=1)
    if highlight:
        a, b, c = highlight
        for i, j in ((a, b), (b, c), (c, a)):
            ax.plot(*zip(pos[i], pos[j]), color="k", lw=1, ls="--", zorder=2)
    col = ["#555555"] * N if S is None else [UP if s > 0 else DOWN for s in S]
    for i in range(N):
        ax.scatter(*pos[i], s=420, color=col[i], zorder=3, edgecolor="k")
        ax.text(*pos[i], str(i) if S is None else ("+" if S[i] > 0 else "-"), ha="center", va="center",
                color="w", fontsize=11, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()


def step02_network(N: int = 8):
    _, P = instance(N, seed=3)
    W = P.W
    tri = _frustrated_triangle(W)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    _draw_net(axes[0], W, highlight=tri, title=f"N = {N} agents; green W_ij > 0 (prefer to agree), "
                                                   f"purple W_ij < 0 (prefer to disagree)")
    i, j, k = tri
    signs = {(a, b): "+" if W[a, b] > 0 else "-" for a, b in ((i, j), (j, k), (k, i))}
    ax = axes[1]
    ax.set_axis_off()
    best = None
    lines = [f"Frustrated triangle ({i}, {j}, {k}) (dashed):",
             f"  sign W_{i}{j} = {signs[(i, j)]},  sign W_{j}{k} = {signs[(j, k)]},  sign W_{k}{i} = {signs[(k, i)]}",
             "  product of the three signs = - :", "", "  satisfied bonds for each of the 8 choices:"]
    for s in all_states(3):
        S3 = dict(zip((i, j, k), s))
        sat = sum(W[a, b] * S3[a] * S3[b] > 0 for a, b in ((i, j), (j, k), (k, i)))
        best = max(best or 0, sat)
        lines.append(f"    S = ({'+' if s[0] > 0 else '-'}, {'+' if s[1] > 0 else '-'}, {'+' if s[2] > 0 else '-'})"
                     f"  ->  {sat} of 3")
    lines += ["", f"  at most {best} of 3: no choice satisfies every pair.",
              "  Frustration is everywhere in SK and is why the",
              "  energy landscape is rugged (steps 5-7)."]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    suptitle(fig, "Step 2 - the agents form a complete network with random-sign couplings: frustration")
    return save(fig, HERE, "step02_network.png")


def step03_configuration(N: int = 8):
    _, P = instance(N, seed=3)
    W = P.W
    rng = np.random.default_rng(5)
    S0 = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    S1, _ = _greedy(W, S0.copy())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    for c, (S, name) in enumerate(((S0, "a random configuration"), (S1, "after steepest descent: a local minimum"))):
        lam = satisfaction(W, S)
        _draw_net(axes[0, c], W, S, title=f"{name}\nH = {energy(W, S):+.3f}, R_N = {-2 * energy(W, S) / N:+.3f}")
        x = np.arange(N)
        axes[1, c].bar(x - 0.2, W @ S, 0.4, color="#888888", label="local field h_i = sum_j W_ij S_j")
        axes[1, c].bar(x + 0.2, lam, 0.4, color=["#2ca02c" if v >= 0 else "#d62728" for v in lam],
                       label="satisfaction lambda_i = S_i h_i")
        axes[1, c].axhline(0, color="k", lw=0.8)
        axes[1, c].set_xticks(x)
        axes[1, c].set_xlabel("agent i")
        axes[1, c].set_title(f"{int(np.sum(lam < 0))} unsatisfied agents (red: lambda_i < 0)", fontsize=10)
        axes[1, c].legend(fontsize=8, loc="lower right")
    suptitle(fig, "Step 3 - local field h_i: what the others push agent i to do; satisfaction lambda_i = S_i h_i;\n"
                  "energy H = -(1/2) sum_i lambda_i, reward R_N = -2H/N")
    return save(fig, HERE, "step03_configuration.png")


def step04_single_flip(N: int = 8):
    _, P = instance(N, seed=3)
    W = P.W
    S = np.where(np.random.default_rng(5).random(N) < 0.5, 1.0, -1.0)
    lam = satisfaction(W, S)
    dH = []
    for k in range(N):
        T = S.copy()
        T[k] = -T[k]
        dH.append(energy(W, T) - energy(W, S))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(N)
    ax.bar(x - 0.2, 2 * lam, 0.4, label="2 lambda_k (predicted)")
    ax.bar(x + 0.2, dH, 0.4, label="H(after flip of k) - H (measured)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("spin k flipped")
    ax.set_ylabel("energy change")
    ax.legend(fontsize=8)
    ax.set_title("Step 4 - single-flip move S_k -> -S_k:  dH_k = 2 lambda_k\n"
                 "a flip lowers H iff agent k is unsatisfied; only the N local fields need updating "
                 "(h_i += 2 W_ik S_k)", fontsize=10)
    return save(fig, HERE, "step04_single_flip.png")


def step05_landscape(N: int = 14):
    _, P = instance(N, seed=7)
    L = landscape(P.W)
    S, E, g = L["S"], L["E"], L["ground"]
    d = hamming_to(S, S[g])
    rng = np.random.default_rng(0)
    jit = d + rng.uniform(-0.3, 0.3, len(d))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(jit, E / N, ".", ms=1.5, color="#bbbbbb", label=f"all 2^{N} = {2 ** N} configurations")
    m = L["is_min"]
    ax.plot(jit[m], E[m] / N, "o", ms=5, color="#d62728", label=f"local minima ({m.sum() // 2} up to S -> -S)")
    ax.plot(d[g], E[g] / N, "*", ms=16, color="k", label="ground state")
    ax.set_xlabel("Hamming distance to the ground state (mod S -> -S)")
    ax.set_ylabel("energy per spin H/N")
    ax.legend(fontsize=8)
    ax.set_title(f"N = {N}: the landscape seen from the ground state", fontsize=10)
    axes[1].hist(E[m] / N, bins=20, color="#d62728", orientation="horizontal")
    axes[1].axhline(E[g] / N, color="k", ls="--", label="ground state")
    axes[1].set_xlabel("number of local minima")
    axes[1].legend(fontsize=8)
    axes[1].set_title("energies of the local minima", fontsize=10)
    suptitle(fig, "Step 5 - the configuration space {-1,+1}^N: a rugged landscape with many local minima "
                  "(lambda_i >= 0 for every i)")
    return save(fig, HERE, "step05_landscape.png")


def step06_basins(N: int = 14):
    _, P = instance(N, seed=7)
    L = landscape(P.W)
    E, basin, g = L["E"], L["basin"], L["ground"]
    mins, size = np.unique(basin, return_counts=True)
    frac = size / len(E)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(E[mins] / N, frac, s=25, color="#d62728")
    gm = [k for k in mins if np.isclose(E[k], E[g])]
    ax.scatter(E[gm] / N, frac[np.isin(mins, gm)], s=160, marker="*", color="k", label="ground state (and its flip)")
    ax.set_xlabel("energy per spin of the local minimum")
    ax.set_ylabel("basin size (fraction of all configurations)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    p_g = frac[np.isin(mins, gm)].sum()
    ax.set_title(f"Step 6 - basins of steepest descent, N = {N}: {len(mins)} minima;\n"
                 f"a random start reaches the ground state with probability {p_g:.2f}", fontsize=10)
    return save(fig, HERE, "step06_basins.png")


def step07_metastable_count(Ns=(6, 8, 10, 12, 14, 16, 18), n_inst: int = 5):
    counts = []
    for N in Ns:
        c = []
        for s in range(n_inst):
            _, P = instance(N, seed=100 + s)
            S = all_states(N)
            lam = S * (S @ P.W)
            c.append(np.all(lam >= -1e-12, axis=1).sum() / 2)
        counts.append(np.mean(c))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(Ns, counts, "o-", label="local minima, exact enumeration (mean of 5)")
    Ns_a = np.array(Ns, float)
    ax.semilogy(Ns_a, counts[-1] * np.exp(0.1992 * (Ns_a - Ns_a[-1])), "--",
                label="slope e^{0.199 N} (Tanaka-Edwards)")
    ax.set_xlabel("N")
    ax.set_ylabel("number of local minima (mod S -> -S)")
    ax.legend(fontsize=8)
    ax.set_title("Step 7 - exponentially many local minima:\nlocal search gets trapped, "
                 "and the trap count grows with N", fontsize=10)
    return save(fig, HERE, "step07_metastable_count.png")


def step08_ground_state(Ns_exact=(8, 12, 16, 20, 24), Ns_pt=(32, 48, 64, 96, 128), n_exact: int = 40, n_pt: int = 10):
    ex = ExactEnumeration(max_N=24)
    pt = ParallelTempering(sweeps=3000, n_replicas=24, beta_min=0.1, beta_max=6.0)
    pts = []
    for N in list(Ns_exact) + list(Ns_pt):
        n = n_exact if N in Ns_exact else n_pt
        e = []
        for s in range(n):
            _, P = instance(N, seed=200 + s)
            r = ex.solve(P).reward if N in Ns_exact else max(pt.solve(P, k).reward for k in range(2))
            e.append(-r / 2)                     # H/N = -R_N / 2
        pts.append((N, np.mean(e), np.std(e) / np.sqrt(n), N in Ns_exact))
    fig, ax = plt.subplots(figsize=(6, 4))
    for exact, mk, lab in ((True, "o", "exact (40 instances)"), (False, "s", "best known, PT (10 instances)")):
        sel = [p for p in pts if p[3] == exact]
        ax.errorbar([p[0] ** (-2 / 3) for p in sel], [p[1] for p in sel], [p[2] for p in sel], fmt=mk, label=lab)
    xx = np.linspace(0, 0.26, 50)
    ax.plot(xx, -PARISI_E0 + 0.75 * xx, "--", color="#888888", label="-0.7632 + 0.75 N^(-2/3) (finite-size law)")
    ax.axhline(-PARISI_E0, color="k", lw=0.8, label=f"Parisi: -{PARISI_E0:.4f}")
    ax.set_xlabel("N^(-2/3)")
    ax.set_ylabel("ground-state energy per spin  min H / N")
    ax.legend(fontsize=8)
    ax.set_title("Step 8 - the optimum we look for: e_0(N) -> -0.7632 as N -> infinity\n"
                 "(maximum reward R_N = -2 e_0 -> 1.526); error bars: standard error over instances", fontsize=10)
    return save(fig, HERE, "step08_ground_state.png")


def step09_game_dynamics(N: int = 48, T: int = 160):
    names = ["ferro_easy", "sg_satisficing", "sg_cycles", "nonreciprocal_chaos", "paramagnet"]
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.4),
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for c, r in enumerate(regimes_by_name(names)):
        J, P = instance(N, r.eps, r.J0, seed=11)
        g = SKGame(GameParams(N=N, alpha=r.alpha, beta=r.beta, eps=r.eps, J0=r.J0), R=1, J=J, seed=4)
        t_eq = int(40 / r.alpha)
        rec = WindowRecorder(0, t_eq + T, store_m=False)
        g.run(t_eq + T, [rec])
        S = rec.result()["S"][0].astype(float)          # (t, N)
        R = np.einsum("ti,ij,tj->t", S, J, S) / N
        axes[0, c].imshow(S[-T:].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axes[0, c].set_title(f"{r.name}\n{r.game_phase}  (alpha={r.alpha:g}, eps={r.eps:g}, "
                             f"T={'0' if r.beta == float('inf') else f'{1 / r.beta:g}'})", fontsize=9)
        axes[0, c].set_xlabel(f"time (last {T} steps)")
        best = ParallelTempering(sweeps=2000).solve(P, 0).reward
        axes[1, c].plot(R, lw=0.8)
        axes[1, c].axhline(best, color="k", ls="--", lw=0.8, label="optimum (best known)")
        axes[1, c].set_xlabel("time t")
        axes[1, c].set_ylim(min(-0.3, R.min()), best * 1.1 + 0.1)
    axes[0, 0].set_ylabel("agent i  (red S=+1, blue S=-1)")
    axes[1, 0].set_ylabel("reward R_N(t)")
    axes[1, 0].legend(fontsize=7)
    suptitle(fig, "Step 9 - the game: each agent plays S_i = sign Q_i, then Q_i <- (1-alpha) Q_i + alpha R_i with "
                  "R_i = sum_j J_ij S_j")
    return save(fig, HERE, "step09_game_dynamics.png")


def step10_satisficing(N: int = 64, n_inst: int = 8, runs: int = 16):
    lam_fp, lam_gs, gap = [], [], []
    for s in range(n_inst):
        J, P = instance(N, seed=300 + s)
        g = SKGame(GameParams(N=N, alpha=0.1), R=runs, J=J, seed=s)
        g.run(3000)
        S = np.where(g.Q >= 0, 1.0, -1.0)
        lam_fp.append((S * (S @ P.W)).ravel())
        best = ParallelTempering(sweeps=3000, n_replicas=24, beta_min=0.1, beta_max=6.0).solve(P, 1)
        Sg = best.S
        lam_gs.append(Sg * (P.W @ Sg))
        gap += list((best.reward - P.reward(S)) / best.reward)
    lam_fp, lam_gs = np.concatenate(lam_fp), np.concatenate(lam_gs)
    rnd = np.where(np.random.default_rng(0).random((200, N)) < 0.5, 1.0, -1.0)
    _, P = instance(N, seed=300)
    lam_rnd = (rnd * (rnd @ P.W)).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-2, 3, 60)
    for lam, lab in ((lam_rnd, "random configuration"), (lam_fp, "game fixed point (alpha=0.1, T=0)"),
                     (lam_gs, "ground state")):
        axes[0].hist(lam, bins=bins, density=True, histtype="step", lw=1.6, label=lab)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("satisfaction lambda_i = S_i h_i")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"fixed points: lambda_i >= 0 for all i (min {lam_fp.min():.3f})\n"
                      "-> every fixed point is a local minimum", fontsize=10)
    axes[1].hist(np.array(gap) * 100, bins=20, color="#ff7f0e")
    axes[1].set_xlabel("satisficing gap: (R_opt - R_fixed point) / R_opt  [%]")
    axes[1].set_title(f"N = {N}: median gap {np.median(gap) * 100:.1f} %, "
                      f"optimum reached in {np.mean(np.array(gap) < 1e-9) * 100:.0f} % of runs", fontsize=10)
    suptitle(fig, "Step 10 - satisficing: at eps = 0 the game stops on a local minimum (everyone satisfied), "
                  "not on the ground state")
    return save(fig, HERE, "step10_satisficing.png")


STEPS = {1: step01_couplings, 2: step02_network, 3: step03_configuration, 4: step04_single_flip,
         5: step05_landscape, 6: step06_basins, 7: step07_metastable_count, 8: step08_ground_state,
         9: step09_game_dynamics, 10: step10_satisficing}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=sorted(STEPS))
    a = ap.parse_args(argv)
    for s in a.steps:
        print("wrote", STEPS[s]())


if __name__ == "__main__":
    main()
