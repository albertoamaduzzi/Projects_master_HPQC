"""Spin state and environment of every algorithm, step by step, on one SK instance (N = 24).

Every figure has the same three rows, sharing the time axis:
  1. SPIN STATE: one row per agent; red = agrees with the ground state S*, blue = disagrees
     (up to the global flip S -> -S, so the ground state is all red);
  2. ENVIRONMENT: what the algorithm reacts to (satisfactions lambda_i, temperature, memory Q_i,
     replicas, relaxed vectors, transverse field ...);
  3. ENERGY H/N, with the ground state dashed.
Vocabulary of ../SK_game/README.md. Traces from traced.py (same update rules as the solvers).

    python state_and_environment.py              # ~1 min
    python state_and_environment.py --steps greedy qa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import instance, plt, save, suptitle  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import traced as T  # noqa: E402
from skgame.optim.solvers import ExactEnumeration  # noqa: E402

N = 24
AGREE = ListedColormap(["#1f77b4", "#d62728"])       # disagree / agree with S*


def _setup():
    _, P = instance(N, seed=21)
    r = ExactEnumeration().solve(P)
    Sg = r.S
    return P.W, Sg, -0.5 * Sg @ P.W @ Sg


def agreement(S_hist, Sg, per_row=False):
    """a_i(t) = S_i(t) S*_i, with ONE global sign for the whole trajectory (chosen so that the final
    state has overlap >= 0 with S*), so that a column never flips colour as a whole.
    per_row=True fixes the sign row by row (used for the 2^N basis states of the quantum figure)."""
    S_hist = np.atleast_2d(S_hist)
    q = S_hist @ Sg
    sgn = np.where(q >= 0, 1.0, -1.0) if per_row else np.full(len(S_hist), 1.0 if q[-1] >= 0 else -1.0)
    return S_hist * Sg[None, :] * sgn[:, None]


def three_rows(title, height_env=1.0, env_extra=False):
    rows = [1.2, height_env, 0.8] if not env_extra else [1.2, 0.35, height_env, 0.8]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.6 * sum(rows) + 0.5), sharex=True,
                             gridspec_kw={"height_ratios": rows, "hspace": 0.12})
    fig.subplots_adjust(top=0.94)
    fig.suptitle(title, y=0.985, fontsize=11)
    return fig, axes


def draw_state(ax, S_hist, Sg, t=None, label="spin state"):
    A = agreement(S_hist, Sg)
    t = np.arange(len(A)) if t is None else t
    ax.imshow(A.T, aspect="auto", cmap=AGREE, vmin=-1, vmax=1, interpolation="nearest",
              extent=(t[0] - 0.5, t[-1] + 0.5, N - 0.5, -0.5))
    frac = (A > 0).mean(1)
    ax.set_ylabel(f"{label}\nagent i\n(red = as in S*)", fontsize=8)
    return frac


def draw_energy(ax, t, E, e0, extra=None):
    ax.plot(t, np.asarray(E) / N, lw=1, color="k", label="energy H/N")
    ax.axhline(e0 / N, color="k", ls="--", lw=0.8, label="ground state")
    if extra:
        for (tt, yy, lab, st) in extra:
            ax.plot(tt, np.asarray(yy) / N, st, lw=1, label=lab)
    ax.set_ylabel("H / N", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")


def draw_lambda(ax, W, S_hist, t=None, marks=None, mark_label="flipped agent"):
    lam = np.array([s * (W @ s) for s in S_hist])
    t = np.arange(len(lam)) if t is None else t
    v = np.abs(lam).max()
    im = ax.imshow(lam.T, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v, interpolation="nearest",
                   extent=(t[0] - 0.5, t[-1] + 0.5, N - 0.5, -0.5))
    if marks is not None:
        ax.plot(np.arange(1, len(marks) + 1), marks, "k.", ms=3, label=mark_label)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("environment:\nsatisfaction lambda_i\n(red < 0 < green)", fontsize=8)
    return im


# ======================================================================= figures =========
def greedy():
    W, Sg, e0 = _setup()
    rng = np.random.default_rng(3)
    tr = T.steepest_descent(W, np.where(rng.random(N) < 0.5, 1.0, -1.0), record=True)
    S_hist = tr["S_hist"]
    moved = [int(np.flatnonzero(S_hist[k] != S_hist[k + 1])[0]) for k in range(len(S_hist) - 1)]
    fig, ax = three_rows("Greedy (steepest descent): at every step the agent with the most negative satisfaction "
                         "flips; it stops when no lambda_i is negative")
    draw_state(ax[0], S_hist, Sg)
    draw_lambda(ax[1], W, S_hist, marks=moved)
    draw_energy(ax[2], np.arange(len(S_hist)), tr["E"], e0)
    ax[2].set_xlabel("flip number")
    return save(fig, HERE, "se_greedy.png")


def game():
    W, Sg, e0 = _setup()
    tr = T.game_learning(W, 0.05, 150, np.random.default_rng(1), record=True)
    fig, ax = three_rows("SK game learning: incentives Q_i integrate the local fields; an agent changes "
                         "decision when its Q_i changes sign; then everything freezes")
    draw_state(ax[0], tr["S_hist"], Sg)
    Q = tr["Q_hist"]
    v = np.abs(Q).max()
    ax[1].imshow(Q.T, aspect="auto", cmap="PuOr", vmin=-v, vmax=v, interpolation="nearest")
    ax[1].set_ylabel("environment:\nincentive Q_i\n(memory of rewards)", fontsize=8)
    draw_energy(ax[2], np.arange(len(tr["E"])), tr["E"], e0)
    ax[2].set_xlabel("time t")
    return save(fig, HERE, "se_game.png")


def sa():
    W, Sg, e0 = _setup()
    rng = np.random.default_rng(2)
    betas = np.geomspace(0.1, 5.0, 60)
    tr = T.simulated_annealing(W, np.where(rng.random(N) < 0.5, 1.0, -1.0), betas, 2, rng, record=True)
    t = np.arange(len(tr["S_hist"]))
    fig, ax = three_rows("Simulated annealing: hot -> the spin state is random (many agents flip although "
                         "satisfied); cooling -> the state freezes, unsatisfied agents disappear", env_extra=True)
    draw_state(ax[0], tr["S_hist"], Sg)
    ax[1].semilogy(t, tr["beta_hist"], color="#9467bd")
    ax[1].set_ylabel("environment:\nbeta = 1/T", fontsize=8)
    draw_lambda(ax[2], W, tr["S_hist"], t=t)
    draw_energy(ax[3], t, tr["E"], e0)
    ax[3].set_xlabel("sweep")
    return save(fig, HERE, "se_sa.png")


def pt():
    W, Sg, e0 = _setup()
    betas = np.geomspace(0.2, 5.0, 12)
    tr = T.parallel_tempering(W, betas, 150, np.random.default_rng(3), record=True)
    Sh = tr["S_hist"]                                   # (sweep, temperature, spin)
    t = np.arange(1, len(Sh) + 1)
    fig, ax = three_rows("Parallel tempering: the coldest replica (top) inherits good configurations from "
                         "hotter ones through swaps; every temperature (middle) keeps exploring")
    draw_state(ax[0], Sh[:, -1, :], Sg, t=t, label="coldest replica")
    q = np.abs(Sh @ Sg) / N                             # |overlap with S*| per temperature
    ax[1].imshow(q.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
                 extent=(t[0] - 0.5, t[-1] + 0.5, len(betas) - 0.5, -0.5))
    ax[1].set_ylabel("environment:\ntemperature index\n(0 hot ... 11 cold)\ncolour = |overlap with S*|", fontsize=8)
    draw_energy(ax[2], t, tr["E"][:, -1], e0, extra=[(t, tr["E"][:, 0], "hottest replica", ":")])
    ax[2].set_xlabel("sweep")
    return save(fig, HERE, "se_pt.png")


def eo():
    W, Sg, e0 = _setup()
    rng = np.random.default_rng(4)
    tr = T.tau_eo(W, np.where(rng.random(N) < 0.5, 1.0, -1.0), 250, 1.3, rng, record=True)
    fig, ax = three_rows("tau-EO: every step flips one agent chosen by its rank in satisfaction (mostly the "
                         "least satisfied); the state never freezes, the best visit is kept")
    draw_state(ax[0], tr["S_hist"], Sg)
    draw_lambda(ax[1], W, tr["S_hist"], marks=tr["moved"])
    draw_energy(ax[2], np.arange(len(tr["E"])), tr["E"], e0,
                extra=[(np.arange(len(tr["E"])), np.minimum.accumulate(tr["E"]), "best so far", "-")])
    ax[2].set_xlabel("step")
    return save(fig, HERE, "se_eo.png")


def tabu():
    W, Sg, e0 = _setup()
    rng = np.random.default_rng(5)
    tenure = 6
    tr = T.tabu_search(W, np.where(rng.random(N) < 0.5, 1.0, -1.0), 120, tenure, record=True)
    fig, ax = three_rows("Tabu search: the least-bad allowed agent flips, even uphill; black dots = agents "
                         f"that are tabu (flipped in the last {tenure} steps)")
    draw_state(ax[0], tr["S_hist"], Sg)
    draw_lambda(ax[1], W, tr["S_hist"])
    ts, ks = [], []
    for t0, k in enumerate(tr["moved"], start=1):
        for d in range(1, tenure + 1):
            if t0 + d <= len(tr["moved"]):
                ts.append(t0 + d)
                ks.append(k)
    ax[1].plot(ts, ks, "k.", ms=2)
    draw_energy(ax[2], np.arange(len(tr["E"])), tr["E"], e0)
    ax[2].set_xlabel("step")
    return save(fig, HERE, "se_tabu.png")


def sdp():
    W, Sg, e0 = _setup()
    rng = np.random.default_rng(6)
    k = int(np.ceil(np.sqrt(2 * N))) + 1
    tr = T.sdp_mixing(W, k, 30, rng, record=True)
    Vh = tr["V_hist"]                                   # (iter, agent, k)
    r = rng.standard_normal(k)                          # one fixed hyperplane
    S_hist = np.where(Vh @ r >= 0, 1.0, -1.0)
    u = np.einsum("i,tik->tk", Sg, Vh)                  # direction of the ground state in vector space
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cos = np.einsum("tik,tk->ti", Vh, u) * Sg[None, :]
    fig, ax = three_rows("SDP + rounding: each agent is a unit vector aligning with its vector local field; "
                         "the spin state is read by a fixed random hyperplane")
    draw_state(ax[0], S_hist, Sg, label="rounded spins")
    ax[1].imshow(cos.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax[1].set_ylabel("environment:\ncos(v_i, ground direction)\n x S*_i", fontsize=8)
    Er = [-0.5 * s @ W @ s for s in S_hist]
    t = np.arange(len(Vh))
    draw_energy(ax[2], t, Er, e0, extra=[(t[1:], tr["relaxed_energy"], "relaxed energy (lower bound)", "--")])
    ax[2].set_xlabel("mixing-method sweep")
    return save(fig, HERE, "se_sdp.png")


def qa(n=8, T_anneal=100.0):
    _, P = instance(n, seed=1)
    W = P.W
    Ez, Sz = T.diag_energies(W)
    g = np.argmin(Ez)
    Sg = Sz[g]
    p_final, hist = T.qa_evolve(W, T_anneal, n_steps=600, record_every=6)
    s = np.array([h[0] for h in hist])
    probs = np.array([h[1] for h in hist])              # (s, 2^n)
    A = agreement(Sz, Sg, per_row=True)                 # (2^n, n), +1 = agent agrees with S*
    agree = probs @ A                                   # expected agreement of each agent
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.5, 1.2, 0.8], "hspace": 0.12})
    fig.subplots_adjust(top=0.94)
    fig.suptitle(f"Quantum annealing (exact, N = {n}, T = {T_anneal:g}): agents start fully undecided "
                  "(all configurations superposed)\nand decide together as the transverse field is removed", y=0.995, fontsize=11)
    im = axes[0].imshow(agree.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                        extent=(0, 1, n - 0.5, -0.5))
    axes[0].set_ylabel("spin state:\n<agreement with S*>\n(0 = undecided)", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.015, pad=0.01)
    axes[1].plot(s, 1 - s, label="A(s): transverse field (keeps agents undecided)")
    axes[1].plot(s, s, label="B(s): SK energy H_P")
    axes[1].set_ylabel("environment:\nschedule", fontsize=8)
    axes[1].legend(fontsize=7)
    levels = np.unique(np.round(Ez, 9))
    M = np.array([[pr[np.isclose(Ez, L)].sum() for L in levels] for pr in probs])
    from matplotlib.colors import LogNorm
    im2 = axes[2].imshow(np.maximum(M.T, 1e-4), aspect="auto", cmap="magma", origin="lower", interpolation="nearest",
                         extent=(0, 1, 0, len(levels)), norm=LogNorm(1e-4, 1))
    fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.01)
    axes[2].set_yticks([0.5, len(levels) - 0.5])
    axes[2].set_yticklabels(["ground", "highest"])
    axes[2].set_ylabel("probability of each\nclassical energy level\n(levels sorted, log colour)", fontsize=8)
    axes[3].plot(s, probs @ Ez / n, "k", lw=1, label="<H_P>/N")
    axes[3].axhline(Ez.min() / n, color="k", ls="--", lw=0.8, label="ground state")
    ax2 = axes[3].twinx()
    ax2.plot(s, probs[:, np.isclose(Ez, Ez.min())].sum(1), color="#2ca02c", label="P(ground state)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(ground)", color="#2ca02c", fontsize=8)
    axes[3].set_ylabel("H / N", fontsize=8)
    axes[3].legend(fontsize=7, loc="center left")
    axes[3].set_xlabel("s = t / T")
    return save(fig, HERE, "se_qa.png")


STEPS = {"greedy": greedy, "game": game, "sa": sa, "pt": pt, "eo": eo, "tabu": tabu, "sdp": sdp, "qa": qa}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    a = ap.parse_args(argv)
    for s in a.steps:
        print("wrote", STEPS[s]())


if __name__ == "__main__":
    main()
