"""Can a quantum annealer represent this problem faithfully? A yes / marginal / no model, no QPU time.

    python main_simulation.py feasibility --topology zephyr --N 16 64 104 --eps 0 --J0 0
    python main_simulation.py feasibility --topology pegasus --N 8:176:8 --sigma 0.02 --beta 5 --plot out.png
    python main_simulation.py feasibility --topology zephyr --N 64 --J-file results/sk/instances/eps0_J00/N0064.npz

Model (all energies in "field units" u = J_rms * sqrt(N-1), the typical field one variable feels from
all the others; J_rms = rms of the problem couplings W_ij of the instance):

 1. Embedding. Clique embedding of K_N on the topology: chain lengths L, and |E_ij| = number of physical
    couplers between chains i and j. N above the largest clique -> NO.
 2. Chains must hold. Chain i is a ferromagnet of strength c in the field h_i = sum_j W_ij S_j; it is
    stable if c >= |h_i| for all i at the low-energy states the anneal reaches. Lower bound:
        p_low = max_i |h_i(S*)| / u ,  S* = best of many greedy local minima  ->  c = p_low * u.
 3. Compression. The hardware range is J in [-1, 1] ([-2, 1] for chains with the extended range, r = 2).
    Everything is divided by D = max(c / r, max |W_ij| / |E_ij|): the programmed logical couplings have
    rms J_rms / D = 1 / (p sqrt(N-1)) * r.
 4. Control error. Each physical coupler is realised with an absolute error sigma (ICE, same units as the
    range). Relative error of a logical coupling:
        eps = sigma * sqrt(|E|) / (J_rms / D)  ~  sigma * sqrt(|E|) * p * sqrt(N-1) / r.
    eps >= 1: the chip holds a different instance (NO); eps <= eps_ok: faithful.
 5. Temperature. The QPU samples roughly P(s) ~ exp(-beta_eff * H_programmed), H_programmed = H / D,
    so the temperature in field units is
        T_u = D / (beta_eff * u) = p / (r * beta_eff).
    It is compared with the problem's own critical temperature T_c (SK: 1; ferromagnetic: J0_eff):
    T_u / T_c >= 1: samples are hot, near-random (NO); <= t_ok: cold enough to reach low energies.

YES = embeddable and eps <= eps_ok and T_u/T_c <= t_ok. These are NECESSARY conditions for the QPU to
be a good sampler of the intended problem, not a promise that it beats classical solvers.
sigma and beta_eff are machine properties: the defaults are typical published orders of magnitude,
not measurements; the output also prints the sigma and beta_eff the problem WOULD need.
"""
from __future__ import annotations

import numpy as np


def _greedy_min(W, S):
    while True:
        h = W @ S
        bad = np.flatnonzero(S * h < 0)
        if bad.size == 0:
            return S
        i = bad[np.argmin(S[bad] * h[bad])]
        S[i] = -S[i]


def field_bound(W: np.ndarray, rng, restarts: int = 30) -> dict:
    """p_low for one instance: largest |field| at the best of `restarts` greedy local minima, in field units."""
    N = len(W)
    jr = float(np.sqrt((W ** 2).sum() / (N * (N - 1))))
    u = jr * np.sqrt(N - 1)
    best = None
    for _ in range(restarts):
        S = _greedy_min(W, rng.choice([-1.0, 1.0], N))
        if best is None or S @ W @ S > best @ W @ best:
            best = S
    h = W @ best
    return {"J_rms": jr, "u": u, "p_low": float(np.abs(h).max() / u), "p_typ": float(np.sqrt(np.mean(h ** 2)) / u),
            "max_abs_W": float(np.abs(W[np.triu_indices(N, 1)]).max())}


def embedding_stats(topology: str, N: int) -> dict | None:
    from skgame.optim.embedding import strategy, topology as topo
    g = topo(topology).ideal_graph()
    st = strategy("clique")
    lim = st.max_size(g)
    if N > lim:
        return {"largest_N": lim, "fits": False}
    emb = st.embed(N, g)
    owner = {q: i for i, ch in emb.items() for q in ch}
    E = {}
    for a, b in g.edges():
        i, j = owner.get(a), owner.get(b)
        if i is not None and j is not None and i != j:
            k = (min(i, j), max(i, j))
            E[k] = E.get(k, 0) + 1
    L = [len(c) for c in emb.values()]
    return {"largest_N": lim, "fits": True, "chain_len_mean": float(np.mean(L)), "chain_len_max": int(max(L)),
            "n_qubits": int(sum(L)), "E_mean": float(np.mean(list(E.values()))), "E_min": int(min(E.values()))}


def assess(topology: str, N: int, W_list, sigma=0.02, beta=5.0, extended=False, eps_ok=0.3, t_ok=0.3,
           T_c=1.0, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    emb = embedding_stats(topology, N)
    row = {"topology": topology, "N": N, "largest_N": emb["largest_N"], "sigma": sigma, "beta_eff": beta,
           "extended_range": extended}
    if not emb["fits"]:
        row.update(verdict="NO", reason=f"N > largest clique {emb['largest_N']} of {topology}")
        return row
    r = 2.0 if extended else 1.0
    fb = [field_bound(W, rng) for W in W_list]
    p = float(np.mean([f["p_low"] for f in fb]))
    u = float(np.mean([f["u"] for f in fb]))
    jr = float(np.mean([f["J_rms"] for f in fb]))
    wmax = float(np.mean([f["max_abs_W"] for f in fb]))
    c = p * u
    D = max(c / r, wmax / 1.0)
    prog = jr / D                                     # rms programmed logical coupling
    eps = sigma * np.sqrt(emb["E_mean"]) / prog
    T_u = D / (beta * u)
    row.update(chain_len_mean=emb["chain_len_mean"], chain_len_max=emb["chain_len_max"], E_mean=emb["E_mean"],
               p_low=p, p_typ=float(np.mean([f["p_typ"] for f in fb])), c_over_Jrms=c / jr,
               c_over_maxW=c / wmax, divisor=D, programmed_rms=prog, eps=float(eps), T_u=float(T_u), T_c=T_c,
               T_ratio=float(T_u / T_c),
               sigma_needed=float(eps_ok * prog / np.sqrt(emb["E_mean"])),
               beta_needed=float(D / (u * t_ok * T_c)))
    if eps >= 1 or T_u / T_c >= 1:
        why = []
        if eps >= 1:
            why.append(f"control error {eps:.2f} x the couplings")
        if T_u / T_c >= 1:
            why.append(f"temperature {T_u / T_c:.2f} x T_c")
        row.update(verdict="NO", reason="; ".join(why))
    elif eps <= eps_ok and T_u / T_c <= t_ok:
        row.update(verdict="YES", reason="faithful and cold")
    else:
        why = []
        if eps > eps_ok:
            why.append(f"control error {eps:.2f} > {eps_ok}")
        if T_u / T_c > t_ok:
            why.append(f"T/T_c {T_u / T_c:.2f} > {t_ok}")
        row.update(verdict="MARGINAL", reason="; ".join(why))
    return row


def instances(N, eps=0.0, J0=0.0, n=8, seed=0, J_file=None):
    from skgame.model import InteractionEnsemble
    if J_file is not None:
        J = np.load(J_file)["J"]
        J = J if J.ndim == 3 else J[None]
    else:
        J = InteractionEnsemble(N, eps, J0=J0).sample(np.random.default_rng([seed, N]), n)
    W = (J + np.swapaxes(J, 1, 2)) / 2
    for w in W:
        np.fill_diagonal(w, 0.0)
    return list(W)


def critical_temperature(eps: float, J0: float) -> float:
    """T_c of the planner's problem in field units: SK spin glass 1; ferromagnet J0_eff (if > 1)."""
    from skgame.optim.regimes import j0_eff
    return max(1.0, j0_eff(eps, J0))


def run(topology, Ns, eps=0.0, J0=0.0, sigma=0.02, beta=5.0, extended=False, eps_ok=0.3, t_ok=0.3,
        n=8, J_file=None, plot=None, log=print) -> list:
    T_c = critical_temperature(eps, J0)
    rows = []
    log(f"[feasibility] {topology}, problem eps={eps:g} J0={J0:g} (T_c = {T_c:g} field units), "
        f"sigma={sigma:g}, beta_eff={beta:g}, extended range={extended}; thresholds eps<={eps_ok}, T/T_c<={t_ok}")
    log(f"[feasibility] {'N':>4s} {'L':>5s} {'p_low':>6s} {'c/Jrms':>7s} {'prog.rms':>9s} {'eps':>6s} "
        f"{'T/Tc':>6s} {'need sigma<=':>12s} {'need beta>=':>11s}  verdict")
    for N in Ns:
        W = instances(N, eps, J0, n=n, J_file=J_file)
        r = assess(topology, N, W, sigma=sigma, beta=beta, extended=extended, eps_ok=eps_ok, t_ok=t_ok, T_c=T_c)
        rows.append(r)
        if "p_low" in r:
            log(f"[feasibility] {N:4d} {r['chain_len_mean']:5.1f} {r['p_low']:6.2f} {r['c_over_Jrms']:7.1f} "
                f"{r['programmed_rms']:9.4f} {r['eps']:6.2f} {r['T_ratio']:6.2f} {r['sigma_needed']:12.4f} "
                f"{r['beta_needed']:11.1f}  {r['verdict']} ({r['reason']})")
        else:
            log(f"[feasibility] {N:4d}  {r['verdict']} ({r['reason']})")
    ok = [r["N"] for r in rows if r["verdict"] == "YES"]
    mid = [r["N"] for r in rows if r["verdict"] in ("YES", "MARGINAL")]
    log(f"[feasibility] largest N with YES: {max(ok) if ok else 'none'}; with YES or MARGINAL: {max(mid) if mid else 'none'}")
    if plot:
        _plot(rows, plot, eps_ok, t_ok)
        log(f"[feasibility] -> {plot}")
    return rows


def _plot(rows, path, eps_ok, t_ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [r for r in rows if "eps" in r]
    if not R:
        return
    x = [r["N"] for r in R]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [r["eps"] for r in R], "o-", label="control error / coupling  (eps)")
    ax.plot(x, [r["T_ratio"] for r in R], "s-", label="temperature / T_c")
    ax.axhline(1, color="r", lw=1, label="NO above")
    ax.axhline(max(eps_ok, t_ok), color="g", lw=1, ls="--", label="YES below (both)")
    for r in R:
        ax.annotate(r["verdict"][0], (r["N"], max(r["eps"], r["T_ratio"])), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("N")
    ax.set_yscale("log")
    ax.set_title(f"{R[0]['topology']}: sigma={R[0]['sigma']:g}, beta_eff={R[0]['beta_eff']:g}, "
                 f"extended={R[0]['extended_range']}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
