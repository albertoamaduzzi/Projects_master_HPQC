"""Figure and numbers for prefactor_and_shimming.tex: where the chain-strength prefactor p must lie.

For SK instances of the campaign (InteractionEnsemble, eps = 0, J_rms = 1/sqrt(N)) it measures, per N:
  * p_typ  = rms local field / (J_rms sqrt(N-1))                      (~1: what a chain feels on average)
  * p_loc  = max_i |h_i| at a greedy local minimum / (J_rms sqrt(N-1))   (the most stressed chain, low energy)
  * p_wc   = max_i sum_j |W_ij| / (J_rms sqrt(N-1))                    (worst case: every neighbour pulls together)
  * c/max|W| for the default p = 1.414 ("chain strength in units of the largest coupling")
and draws the noise ceilings p_noise = r / (sigma sqrt(N-1)), r = 1 (j_range [-1, 1]) or 2 (extended range
[-2, 1] for chains), for an integrated control error sigma of 0.01, 0.02, 0.03 of the coupler range.
Writes prefactor_window.png and prefactor_numbers.tex (macros used by the document).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "unlearnable_games_explanation"))
from skgame.model import InteractionEnsemble  # noqa: E402


def greedy_local_min(W: np.ndarray, rng) -> np.ndarray:
    S = rng.choice([-1.0, 1.0], len(W))
    while True:
        h = W @ S
        bad = np.flatnonzero(S * h < 0)
        if bad.size == 0:
            return S
        i = bad[np.argmin(S[bad] * h[bad])]
        S[i] = -S[i]


def measure(Ns, n_inst=40, seed=3):
    rows = []
    rng = np.random.default_rng(seed)
    for N in Ns:
        J = InteractionEnsemble(N).sample(rng, n_inst)
        W = (J + np.swapaxes(J, 1, 2)) / 2
        typ, loc, wc, cmax = [], [], [], []
        for w in W:
            jr = np.sqrt(np.sum(w ** 2) / (N * (N - 1)))
            u = jr * np.sqrt(N - 1)
            S = greedy_local_min(w, rng)
            h = w @ S
            typ.append(np.sqrt(np.mean(h ** 2)) / u)
            loc.append(np.abs(h).max() / u)
            wc.append(np.abs(w).sum(1).max() / u)
            cmax.append(1.414 * u / np.abs(w).max())
        rows.append({"N": N, "p_typ": np.mean(typ), "p_loc": np.mean(loc), "p_loc_max": np.max(loc),
                     "p_wc": np.mean(wc), "c_over_max_default": np.mean(cmax)})
    return rows


def main():
    Ns = [8, 16, 24, 32, 48, 64, 80, 96, 104, 128, 160, 177]
    rows = measure(Ns)
    for r in rows:
        print(f"N={r['N']:4d}  p_typ={r['p_typ']:.2f}  p_loc={r['p_loc']:.2f} (max {r['p_loc_max']:.2f})  "
              f"p_wc={r['p_wc']:.2f}  sqrt(2/pi)sqrt(N-1)={np.sqrt(2 / np.pi) * np.sqrt(r['N'] - 1):.2f}  "
              f"c/max|W| at p=1.414: {r['c_over_max_default']:.2f}")
    by = {r["N"]: r for r in rows}
    with open(HERE / "prefactor_numbers.tex", "w") as f:
        for N, tag in [(16, "A"), (64, "B"), (104, "C"), (160, "D")]:
            r = by[N]
            for k in ("p_typ", "p_loc", "p_wc", "c_over_max_default"):
                name = "n" + k.replace("_", "").replace("overmax", "OverMax") + tag
                f.write(f"\\newcommand{{\\{name}}}{{{r[k]:.2f}}}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = np.linspace(8, 180, 300)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(Ns, [r["p_typ"] for r in rows], "o-", color="#2a7", label=r"$p_{\rm typ}$: rms local field at a local minimum")
    ax.plot(Ns, [r["p_loc"] for r in rows], "s-", color="#27c", label=r"$p_{\rm loc}$: largest field at a local minimum")
    ax.plot(Ns, [r["p_wc"] for r in rows], "^-", color="#c42", label=r"$p_{\rm wc}$: $\max_i\sum_j|W_{ij}|$ (worst case)")
    ax.plot(n, np.sqrt(2 / np.pi) * np.sqrt(n - 1), ":", color="#c42", lw=1, label=r"$\sqrt{2/\pi}\sqrt{N-1}$")
    for sg, ls in [(0.01, "-"), (0.02, "--"), (0.03, "-.")]:
        ax.plot(n, 1 / (sg * np.sqrt(n - 1)), ls, color="k", lw=1.2,
                label=rf"noise ceiling $1/(\sigma\sqrt{{N-1}})$, $\sigma={sg}$")
        ax.plot(n, 2 / (sg * np.sqrt(n - 1)), ls, color="#888", lw=1.0,
                label="grey: same σ, extended range (×2)" if sg == 0.01 else None)
    for p in (0.75, 1.0, 1.414, 2.0, 3.0, 4.0, 6.0):
        ax.axhline(p, color="#fb0", lw=0.6, zorder=0)
    ax.axvline(104, color="#888", lw=0.8, ls=":")
    ax.text(105, 0.35, "Zephyr\nmax 104", fontsize=8, color="#555")
    ax.axvline(177, color="#888", lw=0.8, ls=":")
    ax.text(160, 0.35, "Pegasus\nmax 177", fontsize=8, color="#555")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 60)
    ax.set_xlim(0, 182)
    ax.set_xlabel("number of agents N")
    ax.set_ylabel(r"chain-strength prefactor $p$   ($c = p\,J_{\rm rms}\sqrt{N-1}$)")
    ax.set_title("Lower bounds (chains hold) vs upper bounds (SK couplings above the control error)\n"
                 "yellow lines: the prefactors tested by coupling-scan", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "prefactor_window.png", dpi=140)
    print("written", HERE / "prefactor_window.png")


if __name__ == "__main__":
    main()
