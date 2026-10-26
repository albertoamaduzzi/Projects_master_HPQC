"""Chain strength, energy compression and noise: the experiments behind README.md.

Part A reproduces, step by step, how the chain strength of the QPU runs is set and what it does
to the SK couplings (measured on real instances and real embeddings). Part B simulates the main
noise sources of an annealer and the procedures that weaken them. Everything here is classical
simulation; no QPU is used, and each figure says which model it assumes.

  A1_field_and_chain_strength   c = p J_rms sqrt(d): J_rms ~ 1/sqrt(N), d = N-1, so c ~ p
  A2_compression_chain          from W_ij to the programmed coupler: split, scale, 0.05
  A3_programmed_couplings       histograms of what the QPU is actually given (Zephyr, Pegasus)
  B1_coupling_noise             random coupler errors change which configuration is the optimum
  B2_relative_noise_vs_N        the same error is ~ sigma p sqrt(N) of a typical coupling
  B3_chain_breaks               chain strength scan on the embedded problem (SA as stand-in)
  B4_temperature                reads at a finite effective temperature: P(ground) vs N
  B5_bias_and_gauges            fixed per-qubit biases and spin-reversal (gauge) averaging

    python noise_and_chains.py            # ~4 min
    python noise_and_chains.py --steps A1 B1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import all_states, energy, instance, plt, save, suptitle  # noqa: E402

import numpy as np  # noqa: E402

from skgame.optim.backends import AnnealBackend, AnnealSettings  # noqa: E402
from skgame.optim.embedding import CliqueEmbedding, topology  # noqa: E402
from skgame.optim.solvers import ExactEnumeration  # noqa: E402

P_DEFAULT = 1.414
CLASSES = [("eps0_J00", 0.0, 0.0), ("eps0_J01", 0.0, 1.0), ("eps0_J03", 0.0, 3.0), ("eps1_J00", 1.0, 0.0)]


def utc(W, p=P_DEFAULT):
    """uniform_torque_compensation on the planner BQM: p * rms(J) * sqrt(mean degree)."""
    N = W.shape[0]
    q = W[np.triu_indices(N, 1)]
    return p * np.sqrt(np.mean(q ** 2)) * np.sqrt(N - 1)


# ===================================================================== part A ==========
def A1_field_and_chain_strength():
    from dwave.embedding.chain_strength import uniform_torque_compensation
    Ns = [16, 32, 64, 96, 128, 160]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, eps, J0 in CLASSES:
        rows = []
        for N in Ns:
            _, P = instance(N, eps, J0, seed=1)
            q = P.W[np.triu_indices(N, 1)]
            c_lib = uniform_torque_compensation(P.to_bqm())          # the library itself
            rows.append((N, np.sqrt(np.mean(q ** 2)), c_lib, utc(P.W)))
        N_, rms, c_lib, c_me = np.array(rows).T
        axes[0].plot(N_, rms * np.sqrt(N_), "o-", label=key)
        axes[1].plot(N_, c_lib, "o-", label=key)
        assert np.allclose(c_lib, c_me), "formula and library disagree"
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("J_rms * sqrt(N)")
    axes[0].set_title("rms coupling ~ (1-eps/2) sqrt(1 + J0^2/N) / sqrt(N)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].axhline(P_DEFAULT, color="k", ls="--", lw=0.8, label="p = 1.414")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("chain strength c")
    axes[1].set_title("c = p J_rms sqrt(N-1) is flat in N:\nthe 1/sqrt(N) and the sqrt(N) cancel", fontsize=9)
    axes[1].legend(fontsize=8)
    N = 160
    _, P = instance(N, seed=1)
    rng = np.random.default_rng(0)
    S = np.where(rng.random((400, N)) < 0.5, 1.0, -1.0)
    h = (S @ P.W).ravel()
    axes[2].hist(h, bins=80, density=True, alpha=0.7, label="local field h_i, random configurations")
    axes[2].axvline(utc(P.W), color="k", ls="--", label=f"c = {utc(P.W):.2f}")
    axes[2].axvline(-utc(P.W), color="k", ls="--")
    axes[2].set_xlabel("h_i = sum_j W_ij S_j")
    axes[2].set_title(f"N = {N}: std(h_i) = {h.std():.3f} = J_rms sqrt(N-1)\n"
                      "c is p times the typical pull of the others on one spin", fontsize=9)
    axes[2].legend(fontsize=8)
    suptitle(fig, "A1 - chain strength = p x (typical local field): for SK it does not depend on N")
    return save(fig, HERE, "A1_field_and_chain_strength.png")


def _embedded_stats(N, topo):
    from dwave.embedding import embed_bqm
    _, P = instance(N, seed=0)
    bqm = P.to_bqm()
    G = topology(topo).ideal_graph()
    emb = CliqueEmbedding().embed(N, G)
    c = utc(P.W)
    t = embed_bqm(bqm, emb, G, chain_strength=c)
    own = {q: i for i, ch in emb.items() for q in ch}
    logical = np.array([v for (a, b), v in t.quadratic.items() if own[a] != own[b]])
    chain = np.array([v for (a, b), v in t.quadratic.items() if own[a] == own[b]])
    lam = 1.0 / max(np.abs(logical).max(), np.abs(chain).max())
    q = P.W[np.triu_indices(N, 1)]
    return dict(J_rms=np.sqrt(np.mean(q ** 2)), c=c, couplers_per_pair=len(logical) / (N * (N - 1) / 2),
                logical_rms=np.sqrt(np.mean(logical ** 2)), lam=lam, logical=logical, chain=chain,
                programmed_rms=np.sqrt(np.mean(logical ** 2)) * lam)


def A2_compression_chain():
    Ns = [16, 32, 64, 96, 128, 160]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for topo, col in (("zephyr", "#1f77b4"), ("pegasus", "#d62728")):
        st = [_embedded_stats(N, topo) for N in Ns]
        ax.plot(Ns, [s["programmed_rms"] for s in st], "o-", color=col, label=f"{topo}: measured")
    Nn = np.array(Ns, float)
    ax.plot(Nn, 1 / (P_DEFAULT * np.sqrt(Nn - 1)), "k--", label="1 / (p sqrt(N-1))  (no splitting)")
    ax.set_xlabel("N")
    ax.set_ylabel("rms of programmed logical couplings (scale [-1, 1])")
    ax.legend(fontsize=8)
    s = _embedded_stats(160, "zephyr")
    ax.set_title(f"N = 160, Zephyr: {s['J_rms']:.4f} (J_rms) x {s['logical_rms'] / s['J_rms']:.3f} (split over "
                 f"{s['couplers_per_pair']:.2f} couplers/pair)\n x {s['lam']:.3f} (= 1/c, auto_scale) = "
                 f"{s['programmed_rms']:.4f}", fontsize=9)
    suptitle(fig, "A2 - the SK information is squeezed into couplings of ~0.05: it falls like 1/sqrt(N)")
    return save(fig, HERE, "A2_compression_chain.png")


def A3_programmed_couplings():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        for N, col in ((64, "#ff7f0e"), (160, "#2ca02c")):
            s = _embedded_stats(N, topo)
            ax.hist(s["logical"] * s["lam"], bins=np.linspace(-0.35, 0.35, 70), histtype="step", lw=1.5, color=col,
                    label=f"N={N}: logical couplers (rms {s['programmed_rms']:.3f})")
        ax.axvline(-1, color="k", lw=2, label="chain couplers: all at -1 after scaling")
        ax.set_xlim(-1.08, 0.4)
        ax.set_yscale("log")
        ax.set_xlabel("programmed coupling J (QPU range [-1, 1])")
        ax.set_title(topo, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    suptitle(fig, "A3 - what the QPU is asked to implement: a wall of chain couplers at -1, "
                  "the whole SK problem in a narrow band around 0")
    return save(fig, HERE, "A3_programmed_couplings.png")


# ===================================================================== part B ==========
def _ground(W):
    S = all_states(W.shape[0])[: 1 << (W.shape[0] - 1)]        # one of each pair S, -S
    E = energy(W, S)
    return S, E


def B1_coupling_noise(Ns=(10, 14, 18), sigmas=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08), n_inst=30):
    """Each programmed coupler gets an independent error N(0, sigma) in QPU units (scale [-1, 1]);
    back in SK units that is sigma * c added to every W_ij. The device then solves W + noise."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rng = np.random.default_rng(0)
    for N in Ns:
        surv, res = [], []
        for sg in sigmas:
            ok, rr = [], []
            for s in range(n_inst):
                _, P = instance(N, seed=500 + s)
                W = P.W
                S, E = _ground(W)
                g = np.argmin(E)
                c = utc(W)
                D = np.triu(rng.normal(0, sg * c, (N, N)), 1)
                Wn = W + D + D.T
                gn = np.argmin(energy(Wn, S))
                ok.append(gn == g)
                rr.append((E[gn] - E[g]) / abs(E[g]))
            surv.append(np.mean(ok))
            res.append(np.mean(rr))
        axes[0].plot(sigmas, surv, "o-", label=f"N = {N}")
        axes[1].plot(sigmas, np.array(res) * 100, "o-", label=f"N = {N}")
    for ax in axes:
        ax.axvspan(0.01, 0.03, color="#dddddd", zorder=0)
        ax.set_xlabel("coupler error sigma (QPU units, range [-1, 1])")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("P(optimum of the noisy problem = true optimum)")
    axes[0].set_title("the noise changes WHICH configuration is optimal\n(grey: order of magnitude often quoted "
                      "for analogue control errors)", fontsize=9)
    axes[1].set_ylabel("residual of the noisy optimum on the true problem [%]")
    axes[1].set_title("even a perfect annealer of the noisy problem\nreturns a slightly wrong answer", fontsize=9)
    suptitle(fig, "B1 - coupler noise (simulated, exact enumeration): the device solves a different problem")
    return save(fig, HERE, "B1_coupling_noise.png")


def B2_relative_noise_vs_N():
    Ns = np.arange(8, 185)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sg in (0.01, 0.02, 0.05):
        ax.plot(Ns, sg * P_DEFAULT * np.sqrt(Ns - 1) * 100, label=f"sigma = {sg}")
    ax.set_xlabel("N")
    ax.set_ylabel("coupler error / typical SK coupling  [%]")
    ax.set_title("error in SK units = sigma c = sigma p J_rms sqrt(N-1);  relative to J_rms: sigma p sqrt(N-1)\n"
                 "N = 160, sigma = 0.02:  0.02 x 1.414 x 12.6 = 0.36  -> 36 % of a typical coupling", fontsize=9)
    ax.legend(fontsize=8)
    suptitle(fig, "B2 - why noise gets worse with N: the chains force the compression 1/(p sqrt(N-1))")
    return save(fig, HERE, "B2_relative_noise_vs_N.png")


def B3_chain_breaks(N=24, prefactors=(0.25, 0.5, 0.75, 1.0, 1.414, 2.0, 3.0), n_inst=4):
    """Chain-strength scan on the embedded problem. The sampler is classical SA on the embedded
    (physical) Ising model of the ideal Zephyr graph: a stand-in that shows the chain effects
    (breaks, freezing), not the quantum dynamics."""
    import tempfile
    out = {m: {"cbf": [], "p": [], "pp": []} for m in ("majority_vote", "minimize_energy")}
    cache = Path(tempfile.mkdtemp())
    for m in out:
        for pf in prefactors:
            cbf, p, pp = [], [], []
            for s in range(n_inst):
                _, P = instance(N, seed=600 + s)
                e0 = ExactEnumeration().solve(P).reward
                be = AnnealBackend("sim:zephyr:clique", AnnealSettings(sim_num_reads=100, sim_num_sweeps=200,
                                   chain_strength_prefactor=pf, chain_break_method=m), cache_root=cache)
                r = be.sample(P, rng=s)
                cnt = np.array(r["counts"])
                cbf.append(r["info"]["chain_break_fraction_mean"])
                p.append(cnt[np.array(r["rewards"]) >= e0 - 1e-9].sum() / cnt.sum())
                pc = np.array(r["polished_counts"])
                pp.append(pc[np.array(r["polished_rewards"]) >= e0 - 1e-9].sum() / pc.sum())
            out[m]["cbf"].append(np.mean(cbf))
            out[m]["p"].append(np.mean(p))
            out[m]["pp"].append(np.mean(pp))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].semilogx(prefactors, out["majority_vote"]["cbf"], "o-")
    axes[0].set_xlabel("chain-strength prefactor p")
    axes[0].set_ylabel("fraction of broken chains")
    axes[0].set_title("weak chains break; p >~ 1 keeps them intact", fontsize=9)
    for m, mk in (("majority_vote", "o"), ("minimize_energy", "s")):
        axes[1].semilogx(prefactors, out[m]["p"], mk + "-", label=f"{m}: raw reads")
        axes[1].semilogx(prefactors, out[m]["pp"], mk + "--", label=f"{m}: + greedy polish")
    axes[1].set_xlabel("chain-strength prefactor p")
    axes[1].set_ylabel("P(read = ground state)")
    axes[1].legend(fontsize=7)
    axes[1].set_title("too weak: broken chains; too strong: rigid chains, the SK part is\n"
                      "relatively weaker -> an optimum in between; polishing repairs a lot", fontsize=9)
    suptitle(fig, f"B3 - chain strength scan, N = {N}, embedded on ideal Zephyr, classical SA as sampler "
                  "(stand-in for the QPU)")
    return save(fig, HERE, "B3_chain_breaks.png")


def B4_temperature(Ns=(8, 10, 12, 14, 16, 18), betas=(2, 5, 10, 20), n_inst=10):
    """Reads modelled as Boltzmann samples of the PROGRAMMED (scaled) problem at inverse temperature
    beta_eff: P(S) ~ exp(-beta_eff H(S) / c). The 1/c is the auto_scale compression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    gaps = []
    for b in betas:
        pg = []
        for N in Ns:
            vals = []
            for s in range(n_inst):
                _, P = instance(N, seed=700 + s)
                S, E = _ground(P.W)
                c = utc(P.W)
                w = np.exp(-b * (E - E.min()) / c)
                vals.append(w.max() / w.sum())
                if b == betas[0]:
                    Es = np.sort(E)
                    gaps.append((N, (Es[1] - Es[0]) / c))
            pg.append(np.mean(vals))
        axes[0].semilogy(Ns, pg, "o-", label=f"beta_eff = {b}")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("P(read = ground state)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("hot reads (beta_eff = 2): the ground state's weight decays with N;\n"
                      "colder: high on average in this small-N range, but not for every instance", fontsize=9)
    g = np.array(gaps)
    for N in Ns:
        sel = g[g[:, 0] == N, 1]
        axes[1].plot(np.full(len(sel), N), sel, ".", color="#888888")
    axes[1].plot(Ns, [np.median(g[g[:, 0] == N, 1]) for N in Ns], "ko-", label="median")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("first excitation energy / c  (programmed units)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("first excitation / c, one dot per instance: some instances are nearly\n"
                      "degenerate (gap ~ 0) -> there any finite temperature mixes the two states", fontsize=9)
    suptitle(fig, "B4 - thermal noise (simulated Boltzmann reads of the programmed problem): what counts "
                  "is the gap compared with T")
    return save(fig, HERE, "B4_temperature.png")


def B5_bias_and_gauges(N=16, sigmas=(0.02, 0.05, 0.1, 0.2), n_gauges=(1, 2, 4, 8, 16), n_inst=40):
    """Systematic error: every qubit carries the same field offset b_i (call after call). A
    spin-reversal transform g in {-1,1}^N programs W_ij g_i g_j and reads S = g * S'; the
    offset then acts as b_i g_i S_i, i.e. with a random sign per transform. Success = the true
    optimum is the optimum for at least one of the transforms used."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    S_all = all_states(N)
    insts = []
    for s in range(n_inst):
        _, P = instance(N, seed=800 + s)
        insts.append((utc(P.W), energy(P.W, S_all)))
    for sg in sigmas:
        hits = np.zeros((n_inst, max(n_gauges)), bool)
        for s, (c, E) in enumerate(insts):
            rng = np.random.default_rng(10_000 * s + int(sg * 1000))
            b = rng.normal(0, sg * c, N)          # ONE fixed offset per qubit: the same in every call
            for t in range(max(n_gauges)):
                gg = np.ones(N) if t == 0 else np.where(rng.random(N) < 0.5, 1.0, -1.0)
                En = E - S_all @ (b * gg)         # the offset enters as b_i g_i S_i
                hits[s, t] = np.isclose(E[np.argmin(En)], E.min())
        ps = [hits[:, :G].any(axis=1).mean() for G in n_gauges]
        ax.plot(n_gauges, ps, "o-", label=f"bias sigma = {sg}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of spin-reversal transforms (gauges)")
    ax.set_ylabel("P(true optimum found by some gauge)")
    ax.legend(fontsize=8)
    ax.set_title(f"N = {N}: a fixed bias always pushes the same way (1 gauge);\n"
                 "random gauges turn it into noise that averages out across calls", fontsize=9)
    suptitle(fig, "B5 - systematic biases and spin-reversal transforms (simulated, exact enumeration)")
    return save(fig, HERE, "B5_bias_and_gauges.png")


STEPS = {"A1": A1_field_and_chain_strength, "A2": A2_compression_chain, "A3": A3_programmed_couplings,
         "B1": B1_coupling_noise, "B2": B2_relative_noise_vs_N, "B3": B3_chain_breaks, "B4": B4_temperature,
         "B5": B5_bias_and_gauges}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    a = ap.parse_args(argv)
    for s in a.steps:
        print("wrote", STEPS[s]())


if __name__ == "__main__":
    main()
