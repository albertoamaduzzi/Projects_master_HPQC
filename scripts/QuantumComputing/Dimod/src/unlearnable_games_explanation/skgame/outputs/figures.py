"""Figure classes: one per figure of the paper (+ two benchmark figures).

Each class is a morphism  make_FigK : Data_1 (x) ... (x) Data_k -> FigK  in the dependency
category; it only reads its inputs and writes figure files in output/figure/<ClassName>/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..registry import FigureOutput
from ..theory.analytic import R_inf, Tc, TAU1
from .common import save_fig, eps_colors, ci95
from . import simulations as D
from . import theory as T
from . import benchmark as B

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 110})


def _panel(ax, lab):
    ax.text(-0.14, 1.04, f"({lab})", transform=ax.transAxes, fontsize=10, fontweight="bold")


class Fig01PhaseDiagram(FigureOutput):
    depends_on = (D.PhaseClassification, T.AnalyticCurves)
    paper_ref = "Fig. 1"
    description = "Schematic (eps, alpha) phase diagrams (noiseless, weak noise) + numerical classification."

    def compute(self, deps):
        pc = deps["PhaseClassification"]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        # (a) schematic noiseless (boundaries as in the paper)
        ax = axs[0]
        ax.fill_between([0, 0.8], 0, 1, color="#cfe2ff")
        ax.fill_between([0.8, 1.2], 0, 1, color="#e9ecef")
        ax.fill_between([1.2, 2.0], 0, 1, color="#d1c4e9")
        ax.text(0.4, 0.8, "L=2\n(Q)cycle", ha="center"); ax.text(1.6, 0.8, "L=4\n(Q)cycle", ha="center")
        ax.text(0.4, 0.2, "(Q)FP", ha="center"); ax.text(1.0, 0.5, "chaos", rotation=90, ha="center", va="center")
        ax.text(1.6, 0.4, r"$L\sim\alpha^{-1/2}$", ha="center")
        ax.set(xlim=(0, 2), ylim=(0, 1), xlabel=r"$\varepsilon$", ylabel=r"$\alpha$", title=r"$\beta\to\infty$ (schematic)")
        # (b) schematic weak noise
        ax = axs[1]
        for x0, x1, c, lab in [(0, 0.47, "#d1e7dd", "aging"), (0.47, 1.2, "#e9ecef", "chaos"),
                               (1.2, 1.85, "#fff3cd", "oscil."), (1.85, 2, "#f8d7da", "random")]:
            ax.fill_between([x0, x1], 0, 1, color=c)
            ax.text((x0 + x1) / 2, 0.15, lab, ha="center", rotation=90 if lab == "random" else 0)
        ax.text(1.0, 0.75, "noisy chaos\n$q>0$", ha="center")
        ax.set(xlim=(0, 2), ylim=(0, 1), xlabel=r"$\varepsilon$", title=r"$\beta\gg1$ (schematic)")
        # (c) numerical: dominant class
        ax = axs[2]
        frac = pc["frac"]
        dom = np.argmax(frac, axis=-1)
        cm = matplotlib.colors.ListedColormap(["#cfe2ff", "#9ec5fe", "#d1c4e9", "#adb5bd"])
        ax.pcolormesh(pc["eps"], pc["alphas"], dom, cmap=cm, vmin=-0.5, vmax=3.5, shading="nearest")
        for k, lab in enumerate(["FP", "L=2", "L=4", "no recurrence"]):
            ax.plot([], [], "s", color=cm(k), label=lab)
        ax.legend(fontsize=7, loc="upper center", ncol=2)
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"$\alpha$", title=r"numerical, finite $N$, $\beta=\infty$")
        return save_fig(fig, self.folder(), "fig01_phase_diagram")


class Fig02AverageReward(FigureOutput):
    depends_on = (D.RewardScanAlpha, D.RewardScanEps, D.RewardScanNoise, D.RewardScanN, T.AnalyticCurves)
    paper_ref = "Fig. 2"
    description = "Average reward vs alpha, eps, noise 1/beta and N."

    def compute(self, deps):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        a = deps["RewardScanAlpha"]
        ax = axs[0, 0]
        for e, c, R in zip(a["eps"], eps_colors(a["eps"]), a["reward"]):
            ax.errorbar(a["alphas"], R.mean(-1), ci95(R, -1), fmt="o-", ms=3, color=c, mfc="none")
            ax.axhline(R_inf(e), ls="--", color=c, lw=0.7)
        ax.set(xlabel=r"$\alpha$", ylabel=r"$\mathcal{R}_N$"); _panel(ax, "a")
        b = deps["RewardScanEps"]
        ax = axs[0, 1]
        for al, R, mk in zip(b["alphas"], b["reward"], ["o", "D"]):
            ax.errorbar(b["eps"], R.mean(-1), ci95(R, -1), fmt=mk, ms=3, mfc="none", label=fr"$\alpha={al}$")
        ee = np.linspace(0, 2, 50)
        ax.plot(ee, R_inf(ee), "k--", lw=0.8, label=r"$\mathcal{R}_\infty$")
        ax.axvline(0.8, ls=":", color="gray")
        ax.set(xlabel=r"$\varepsilon$"); ax.legend(fontsize=7); _panel(ax, "b")
        c_ = deps["RewardScanNoise"]
        ax = axs[1, 0]
        for e, col, R in zip(c_["eps"], ["#3b0f70", "#2c7fb8"], c_["reward"]):
            ax.errorbar(c_["T"], R.mean(-1), ci95(R, -1), fmt="o-", ms=3, mfc="none", color=col, label=fr"$\varepsilon={e}$")
            ax.axhline(R_inf(e), ls="--", color=col, lw=0.7)
        ax.set(xlabel=r"$1/\beta$", ylabel=r"$\mathcal{R}_N$"); ax.legend(fontsize=7); _panel(ax, "c")
        d = deps["RewardScanN"]
        ax = axs[1, 1]
        x = d["Ns"] ** (-2 / 3)
        xx = np.linspace(0, x.max() * 1.05, 20)
        for e, col, R, (R0, A) in zip(d["eps"], ["#3b0f70", "#2c7fb8"], d["reward"], d["fit"]):
            ax.errorbar(x, R.mean(-1), ci95(R, -1), fmt="o", ms=3, mfc="none", color=col)
            ax.plot(xx, R0 - A * xx, color=col, lw=0.8, label=fr"$\varepsilon={e}$: $A\approx{A:.2f}$")
        ax.plot(xx, R_inf(0) - 1.5 * xx, "k:", lw=0.8, label="SK ground state, A≈1.5")
        ax.set(xlabel=r"$N^{-2/3}$"); ax.legend(fontsize=7); _panel(ax, "d")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig02_average_reward")


class Fig03Autocorrelation(FigureOutput):
    depends_on = (D.AutocorrAlpha1, T.DMFTAlpha1)
    paper_ref = "Fig. 3"
    description = "Steady-state C(tau) at alpha=1: L=2 cycles, weak chaos, strong chaos, L=4 cycles; sims vs DMFT."

    def compute(self, deps):
        s, t = deps["AutocorrAlpha1"], deps["DMFTAlpha1"]
        fig, axs = plt.subplots(2, 2, figsize=(7, 5))
        for k, ax in enumerate(axs.ravel()):
            C = s["C3"][k]
            tau = np.arange(C.shape[1])
            ax.errorbar(tau, C.mean(0), ci95(C, 0), fmt="o", mfc="none", color="k", ms=4)
            ax.plot(tau, t["C_fig3"][k], "-", color="k", lw=1)
            ax.set_title(fr"$\varepsilon={s['eps3'][k]}$"); _panel(ax, "abcd"[k])
            ax.set(xlabel=r"$\tau$", ylabel=r"$C(\tau)$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig03_autocorrelation_alpha1")


class Fig04RewardDistribution(FigureOutput):
    depends_on = (D.RewardDistributions,)
    paper_ref = "Fig. 4"
    description = "rho(e_i) at the fixed points (eps=0) for three alpha; linear vanishing at e->0."

    def compute(self, deps):
        d = deps["RewardDistributions"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bins = np.linspace(0, 5, 41)
        cols = ["#1b4332", "#40916c", "#95d5b2"]
        for a, e, c in zip(d["alphas"], d["e_fig4"], cols):
            h, _ = np.histogram(e, bins=bins, density=True)
            ax.plot(0.5 * (bins[1:] + bins[:-1]), h, "D", mfc="none", color=c, ms=4, label=fr"$\alpha={a}$")
        x = np.linspace(0, 0.8, 10)
        ax.plot(x, 1.6 * x, "k--", lw=0.8, label=r"$\kappa e$, $\kappa=1.6$")
        ax.plot(x, 0.6 * x, "k:", lw=0.8, label=r"$\kappa_{SK}\approx0.6$")
        ax.set(xlabel=r"$e_i$", ylabel=r"$\rho(e_i)$", xlim=(0, 5), ylim=(0, 0.75)); ax.legend(fontsize=7, loc="center left")
        ins = ax.inset_axes([0.62, 0.45, 0.35, 0.45])
        allE = np.concatenate([np.ravel(e) for e in d["e_fig4"]])
        g = np.linspace(2, 5.5, 40)
        ins.semilogy(g, [np.mean(allE > v) for v in g], ".", ms=3)
        ins.set_title(r"$\rho^>(e)$", fontsize=7)
        return save_fig(fig, self.folder(), "fig04_reward_distribution")


class Fig05RewardTrajectories(FigureOutput):
    depends_on = (D.RewardTrajectories,)
    paper_ref = "Fig. 5"
    description = "Individual rewards e_i(t) on a log time axis, histograms at t=1 and final time."

    def compute(self, deps):
        d = deps["RewardTrajectories"]
        fig, axs = plt.subplots(2, 2, figsize=(8, 4.5), gridspec_kw={"width_ratios": [4, 1]})
        keep = d["t"] >= 1
        x = d["alpha"] * d["t"][keep]
        for k in range(2):
            E = d["e"][k][keep]
            cm = plt.get_cmap("Blues")
            for i in range(0, E.shape[1], 4):
                axs[k, 0].plot(x, E[:, i], lw=0.5, color=cm(0.3 + 0.7 * i / E.shape[1]))
            axs[k, 0].set_xscale("log"); axs[k, 0].set_ylim(-3, 3.5)
            axs[k, 0].set(ylabel=r"$e_i(t)$", title=fr"$\varepsilon={d['eps'][k]}$")
            axs[k, 1].hist(E[1], bins=30, orientation="horizontal", color="gray", alpha=0.5, density=True)
            axs[k, 1].hist(E[-1], bins=30, orientation="horizontal", histtype="step", color="k", density=True)
            axs[k, 1].set_ylim(-3, 3.5)
        axs[1, 0].set_xlabel(r"$\alpha t$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig05_reward_trajectories")


class Fig06RewardDistributionsEps(FigureOutput):
    depends_on = (D.RewardDistributions,)
    paper_ref = "Fig. 6"
    description = "rho(e_i) for alpha in {1,0.5,0.01} across eps: negative tail grows with competition."

    def compute(self, deps):
        d = deps["RewardDistributions"]
        x = 0.5 * (d["bins"][1:] + d["bins"][:-1])
        fig, axs = plt.subplots(1, len(d["eps"]), figsize=(14, 2.6), sharey=True)
        for j, ax in enumerate(axs):
            for i, (mk, c) in enumerate(zip(["v", "s", "o"], ["k", "#7b2cbf", "#52b788"])):
                h = d["hist"][i, j]
                ax.semilogy(x[h > 0], h[h > 0], mk, mfc="none", ms=3, color=c, label=fr"$\alpha={d['alphas'][i]}$")
            ax.set(title=fr"$\varepsilon={d['eps'][j]}$", xlabel=r"$e_i$", ylim=(1e-4, 2))
        axs[0].set_ylabel(r"$\rho(e_i)$"); axs[0].legend(fontsize=6)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig06_reward_distributions_eps")


class Fig07CrossOverlap(FigureOutput):
    depends_on = (D.CrossOverlap,)
    paper_ref = "Fig. 7"
    description = "Overlap of final rewards between two initial conditions: distribution (N^{2/3} scaling) and mean ~ N^-0.85."

    def compute(self, deps):
        d = deps["CrossOverlap"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        for N, c in zip(d["Ns"], d["Cx"]):
            z = N ** (2 / 3) * (c - c.mean())
            h, b = np.histogram(z, bins=8, density=True)
            axs[0].plot(0.5 * (b[1:] + b[:-1]), h / N ** (2 / 3), "o-", ms=3, mfc="none", label=f"N={N}")
        axs[0].set(xlabel=r"$N^{2/3}(C^\times_N-\overline{C^\times_N})$", ylabel=r"$\rho(C^\times_N)/N^{2/3}$"); axs[0].legend(fontsize=7)
        m = d["Cx"].mean(1)
        axs[1].errorbar(1 / d["Ns"], m, ci95(d["Cx"], 1), fmt="o", mfc="none")
        sel = m > 0
        if sel.sum() >= 2:
            sl, ic = np.polyfit(np.log(1 / d["Ns"][sel]), np.log(m[sel]), 1)
            xx = np.geomspace((1 / d["Ns"]).min(), (1 / d["Ns"]).max(), 10)
            axs[1].plot(xx, np.exp(ic) * xx**sl, "k--", label=fr"$\propto N^{{-{sl:.2f}}}$ (paper: 0.85)")
        axs[1].set(xscale="log", yscale="log", xlabel="1/N", ylabel=r"$\overline{C^\times_N}$"); axs[1].legend(fontsize=7)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig07_cross_overlap")


class Fig08Cooperation(FigureOutput):
    depends_on = (D.CooperationScan,)
    paper_ref = "Fig. 8"
    description = "Reward and |M| vs cooperative incentive J0."

    def compute(self, deps):
        d = deps["CooperationScan"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        for e, c, R, M in zip(d["eps"], eps_colors(d["eps"]), d["reward"], d["M"]):
            axs[0].errorbar(d["J0"], R.mean(-1), ci95(R, -1), fmt="o", ms=3, mfc="none", color=c)
            axs[1].errorbar(d["J0"], M.mean(-1), ci95(M, -1), fmt="o", ms=3, mfc="none", color=c, label=fr"$\varepsilon={e}$")
        axs[0].plot([0, 2], [0, 2], "k-", lw=0.8)
        axs[0].set(xlabel=r"$J_0$", ylabel=r"$\mathcal{R}_N$"); axs[1].set(xlabel=r"$J_0$", ylabel=r"$|M|$")
        axs[1].legend(fontsize=6)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig08_cooperation")


class Fig09Complexity(FigureOutput):
    depends_on = (D.FixedPointCounts, T.AnalyticCurves)
    paper_ref = "Fig. 9"
    description = "Annealed complexity of NMFE fixed points: (T, eps) heat map and T=0 curve vs eps."

    def compute(self, deps):
        d, a = deps["FixedPointCounts"], deps["AnalyticCurves"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        ax = axs[0]
        im = ax.pcolormesh(d["eps_T"], d["T_over_2"], np.maximum(d["Sigma_T"], 1e-3), shading="nearest",
                           cmap="magma", norm=matplotlib.colors.LogNorm(1e-3, 0.25))
        ax.plot(a["eps"], a["Tc"] / 2, "w--"); ax.axvline(1, color="w")
        ax.set(xlim=(d["eps_T"].min(), d["eps_T"].max()), ylim=(0, 1.05), xlabel=r"$\varepsilon$", ylabel=r"$T/2$",
               title=f"multistart lower bound, N={d['N_T']}")
        fig.colorbar(im, ax=ax, label=r"$\Sigma$"); _panel(ax, "a")
        ax = axs[1]
        ax.plot(a["eps_le1"], a["Sigma_fp"], "k-", label=r"Eq. (22), $\beta\to\infty$")
        ax.plot(d["eps_exact"], d["Sigma_exact"], "o", mfc="none", label=f"exact enumeration, N={d['N_exact']}")
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"$\Sigma(\varepsilon)$"); ax.legend(fontsize=7); _panel(ax, "b")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig09_fixed_point_complexity")


class Fig10CycleComplexity(FigureOutput):
    depends_on = (T.CycleComplexity,)
    paper_ref = "Fig. 10"
    description = "Sigma_2(alpha, eps) and 1-C(1) at the L=2 saddle: coalescence with the fixed-point saddle as alpha->0."

    def compute(self, deps):
        d = deps["CycleComplexity"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        for e, c, S, oc, Sf in zip(d["eps"], eps_colors(d["eps"]), d["Sigma2"], d["one_minus_C1"], d["Sigma_fp"]):
            axs[0].plot(d["alpha"], S, color=c, label=fr"$\varepsilon={e}$")
            axs[0].axhline(Sf, ls="--", color=c, lw=0.8)
            axs[1].plot(d["alpha"], oc, color=c)
        axs[0].set(xlabel=r"$\alpha$", ylabel=r"$\Sigma_2(\alpha,\varepsilon)$"); axs[0].legend(fontsize=7)
        axs[1].set(xlabel=r"$\alpha$", ylabel=r"$1-C(\tau=1)$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig10_cycle_complexity")


class Fig11C2vsEps(FigureOutput):
    depends_on = (D.AutocorrAlpha1, T.DMFTAlpha1)
    paper_ref = "Fig. 11"
    description = "C(tau=2) at alpha=1 vs eps for several N, against DMFT (N->inf first)."

    def compute(self, deps):
        s, t = deps["AutocorrAlpha1"], deps["DMFTAlpha1"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for N, C, mk in zip(s["Ns11"], s["C11"], ["s", "v", "o", "D", "^"]):
            ax.errorbar(s["eps11"], C.mean(-1), ci95(C, -1), fmt=mk, ms=4, mfc="none", label=f"N={N}")
        ax.plot(t["eps_fig11"], t["C2_fig11"], "k-x", label="DMFT")
        ax.axvline(0.8, ls=":", color="gray")
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"$C(\tau=2)$"); ax.legend(fontsize=7)
        return save_fig(fig, self.folder(), "fig11_C2_vs_eps")


class Fig12FrozenVsAlpha(FigureOutput):
    depends_on = (D.FrozenVsAlpha,)
    paper_ref = "Fig. 12"
    description = "C(tau=1) vs alpha (eps<0.8): memory turns 2-cycles into fixed points; sample trajectories."

    def compute(self, deps):
        d = deps["FrozenVsAlpha"]
        fig = plt.figure(figsize=(9, 3.4))
        ax = fig.add_subplot(1, 2, 1)
        for e, c, C in zip(d["eps"], eps_colors(d["eps"]), d["C1"][-1]):
            ax.errorbar(d["alphas"], C.mean(-1), ci95(C, -1), fmt="o", ms=3, mfc="none", color=c, label=fr"$\varepsilon={e}$")
        if len(d["Ns"]) > 1:
            for e, c, C in zip(d["eps"], eps_colors(d["eps"]), d["C1"][0]):
                ax.plot(d["alphas"], C.mean(-1), "v", ms=3, mfc="none", color=c)
        ax.set(xlabel=r"$\alpha$", ylabel=r"$C(\tau=1)$"); ax.legend(fontsize=6)
        for k in range(3):
            a2 = fig.add_subplot(3, 2, 2 * (k + 1))
            a2.imshow(d["traj"][k].T, aspect="auto", cmap="binary", interpolation="nearest")
            a2.set_yticks([]); a2.set_ylabel(fr"$\alpha={d['traj_alphas'][k]}$", fontsize=7)
            if k < 2:
                a2.set_xticks([])
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig12_frozen_vs_alpha")


class Fig13MemoryCorrelation(FigureOutput):
    depends_on = (D.MemoryCorrelation, T.DMFTMemory)
    paper_ref = "Fig. 13"
    description = "C(tau=2/alpha) vs eps for several alpha, N; DMFT lines; trajectories at eps=1."

    def compute(self, deps):
        s, t = deps["MemoryCorrelation"], deps["DMFTMemory"]
        fig = plt.figure(figsize=(10, 3.6))
        ax = fig.add_subplot(1, 2, 1)
        cm = plt.get_cmap("viridis_r")
        amin, amax = np.log(min(s["alphas"])), np.log(max(s["alphas"]))
        for k, a in enumerate(s["alphas"]):
            c = cm((np.log(a) - amin) / max(amax - amin, 1e-9))
            for i, (N, mk) in enumerate(zip(s["Ns"], "vod")):
                C = s["C"][i, k]
                ax.errorbar(s["eps"], C.mean(-1), ci95(C, -1), fmt=mk, ms=3, mfc="none", color=c)
            if a in t["alphas"]:
                ax.plot(t["eps"], t["C_2_over_alpha"][list(t["alphas"]).index(a)], "-x", color=c, label=fr"DMFT $\alpha={a}$")
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"$C(\tau=2/\alpha)$"); ax.legend(fontsize=6)
        for k in range(2):
            a2 = fig.add_subplot(2, 2, 2 * (k + 1))
            a2.imshow(s[f"traj_{k}"], aspect="auto", cmap="binary", interpolation="nearest")
            a2.set_yticks([]); a2.set_ylabel(fr"$\alpha={s['traj_alphas'][k]}$", fontsize=7)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig13_memory_correlation")


class Fig14Eps1Decorrelation(FigureOutput):
    depends_on = (D.Eps1Decorrelation, T.DMFTMemory, T.AnalyticCurves)
    paper_ref = "Fig. 14"
    description = "C(tau) at eps=1: collapse in alpha*tau onto the Sompolinsky-Crisanti solution, tau1=2.84."

    def compute(self, deps):
        s, t, a = deps["Eps1Decorrelation"], deps["DMFTMemory"], deps["AnalyticCurves"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        cm = plt.get_cmap("viridis_r")
        for k, al in enumerate(s["alphas"]):
            c = cm(k / max(len(s["alphas"]) - 1, 1))
            raw = s[f"C_raw_0_{k}"]
            axs[0].plot(np.arange(len(raw)), raw, "o", ms=2, color=c, label=fr"$\alpha={al:.2g}$")
            axs[1].semilogy(s["alpha_tau"], s["C"][0, k], "o", ms=2, color=c)
        for k, al in enumerate(t["alphas_eps1"]):
            axs[1].semilogy(t["alpha_tau"], t["C_eps1"][k], "-", lw=0.6, color="gray")
        axs[1].semilogy(a["sc_tau"], a["sc_C"], "k-", lw=1.2, label="Sompolinsky-Crisanti")
        axs[1].semilogy(a["sc_tau"], (2 / np.pi) * np.exp(-a["sc_tau"] / TAU1), "k--", lw=0.8, label=fr"$e^{{-\tau/\tau_1}}$, $\tau_1={TAU1:.2f}$")
        axs[0].set(xlim=(0, 100), xlabel=r"$\tau$", ylabel=r"$C(\tau)$"); axs[0].legend(fontsize=6)
        axs[1].set(xlim=(0, 8), ylim=(1e-2, 1.1), xlabel=r"$\alpha\tau$"); axs[1].legend(fontsize=6)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig14_eps1_decorrelation")


class Fig15CycleStretching(FigureOutput):
    depends_on = (D.CyclePeriod, T.DMFTMemory)
    paper_ref = "Fig. 15"
    description = "Anomalous stretching of cycles: 1/omega0 ~ alpha^{-1/2}; C vs sqrt(alpha) tau; power spectrum."

    def compute(self, deps):
        s, t = deps["CyclePeriod"], deps["DMFTMemory"]
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(2, 1, 1)
        cm = plt.get_cmap("viridis_r")
        for e, w in zip(s["eps"], s["omega0"]):
            ax.loglog(1 / s["alphas"], 1 / w, "o", ms=4, mfc="none", color=cm((e - 1.2) / 0.8), label=fr"$\varepsilon={e}$")
        ax.loglog(1 / t["alphas_cycles"], 1 / t["omega0"], "kx", label=fr"DMFT $\varepsilon={t['eps_cycles']}$")
        x = np.array([3, 30.0])
        ok = np.isfinite(s["omega0"]) & (s["eps"][:, None] >= 1.4)
        c0 = np.nanmedian((1 / s["omega0"])[ok] / np.sqrt(1 / np.broadcast_to(s["alphas"], s["omega0"].shape)[ok]))
        ax.loglog(x, 0.6 * c0 * np.sqrt(x), "k--", lw=0.8, label=r"$\propto\alpha^{-1/2}$")
        ax.set(xlabel=r"$1/\alpha$", ylabel=r"$1/\omega_0$  ($\omega$ in cycles per step)"); ax.legend(fontsize=6, ncol=2); _panel(ax, "a")
        ax = fig.add_subplot(2, 2, 3)
        for k, (al, C) in enumerate(zip(s["alphas_C"], s["C_eps"])):
            ax.plot(s["sqrt_alpha_tau"], C, color=plt.get_cmap("Greens")(0.4 + 0.6 * k / 4), label=fr"$\alpha={al:.2g}$")
        ax.set(xlabel=r"$\sqrt{\alpha}\tau$", ylabel=r"$C(\tau)$"); ax.legend(fontsize=6); _panel(ax, "b")
        ax = fig.add_subplot(2, 2, 4)
        for k, P in enumerate(s["power"]):
            ax.semilogy(s["w_over_sqrt_alpha"][1:], P[1:], color=plt.get_cmap("Greens")(0.4 + 0.6 * k / 4))
        ax.set(xlabel=r"$\omega/\sqrt{\alpha}$", ylabel=r"$\alpha|\hat C(\omega)|^2$"); _panel(ax, "c")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig15_cycle_stretching")


class Fig16QHeatmap(FigureOutput):
    depends_on = (D.QHeatmap, T.AnalyticCurves)
    paper_ref = "Fig. 16"
    description = "q in the (eps, T) plane for alpha in {0.5, 0.1, 0.01} with T_c(eps)."

    def compute(self, deps):
        d, a = deps["QHeatmap"], deps["AnalyticCurves"]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        for k, ax in enumerate(axs):
            im = ax.pcolormesh(d["eps"], d["T_over_2"], d["q"][k], shading="nearest", cmap="inferno", vmin=0, vmax=1)
            ax.plot(a["eps"], a["Tc"] / 2, "w--")
            ax.set(ylim=(0, 1), xlabel=r"$\varepsilon$", title=fr"$\alpha={d['alphas'][k]}$")
        axs[0].set_ylabel(r"$T/2$")
        fig.colorbar(im, ax=axs, label="q")
        return save_fig(fig, self.folder(), "fig16_q_heatmap")


class Fig17OrderParameter(FigureOutput):
    depends_on = (D.QvsTemperature, T.StaticDMFTSolution, T.DMFTFiniteBeta)
    paper_ref = "Fig. 17"
    description = "q vs T/T_c: simulations, dynamical DMFT and the (incorrect) static RS solution; spectral instability."

    def compute(self, deps):
        s, st, dy = deps["QvsTemperature"], deps["StaticDMFTSolution"], deps["DMFTFiniteBeta"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        for a, q, mk in zip(s["alphas"], s["q"], "oD"):
            axs[0].plot(s["x_T"], q, mk, mfc="none", ms=4, label=fr"sim $\alpha={a}$")
        axs[0].plot(dy["x_T"], dy["q"], "kx", label="DMFT (dynamic)")
        axs[0].plot(st["T_over_Tc"], st["q"], "-", color="#3b0f70", label="static DMFT / RS")
        axs[0].set(xlabel=r"$T/T_c(\varepsilon)$", ylabel="q"); axs[0].legend(fontsize=6)
        axs[1].plot(st["x_spec"], st["inv_spec"], color="#3b0f70")
        axs[1].axhline(0, ls=":", color="k"); axs[1].axhspan(-1, 0, color="gray", alpha=0.3)
        axs[1].set(ylim=(-0.25, max(0.45, np.nanmax(st["inv_spec"]))), xlabel=r"$T/T_c(\varepsilon)$",
                   ylabel=r"$\langle|\hat Q_1(\omega=0)|^2\rangle^{-1}$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig17_order_parameter")


class Fig18Aging(FigureOutput):
    depends_on = (D.Aging,)
    paper_ref = "Fig. 18"
    description = "Aging: C(t_w,t_w+t) vs t/t_w (eps small: collapse = simple aging); vs alpha t for eps=0.5 (TTI)."

    def compute(self, deps):
        d = deps["Aging"]
        fig, axs = plt.subplots(1, 4, figsize=(14, 3))
        cm = plt.get_cmap("inferno")
        for s, ax in enumerate(axs):
            a, b, e = d["sets"][s]
            C, lags, tws = d[f"C_{s}"], d[f"lags_{s}"], d[f"tw_{s}"]
            if s < 2:
                for k, tw in enumerate(tws):
                    ax.semilogx(lags / tw, C[k], "-", color=cm(0.15 + 0.7 * k / len(tws)))
                ax.set(xlabel=r"$t/t_w$", title=fr"$\alpha={a},\beta={b},\varepsilon={e}$")
            else:
                for k, tw in enumerate(tws):
                    ax.semilogx(a * lags, C[k], "-", color=cm(0.15 + 0.7 * k / len(tws)))
                ax.set(xlabel=r"$\alpha t$", title=fr"$\alpha={a},\beta={b},\varepsilon={e}$")
        axs[0].set_ylabel(r"$C(t_w,t_w+t)$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig18_aging")


class Fig19FiniteBeta(FigureOutput):
    depends_on = (D.FiniteBetaTrajectories, T.DMFTFiniteBeta)
    paper_ref = "Fig. 19"
    description = "Finite-beta trajectories m_i(t), their histogram and C(tau) (sims vs DMFT)."

    def compute(self, deps):
        d, t = deps["FiniteBetaTrajectories"], deps["DMFTFiniteBeta"]
        nb, ne = len(d["betas"]), len(d["eps"])
        fig, axs = plt.subplots(ne, 3 * nb, figsize=(14, 9), gridspec_kw={"width_ratios": [4, 1, 3] * nb})
        x = d["alpha"] * np.arange(d["traj"].shape[-1])
        mid = 0.5 * (d["hist_edges"][1:] + d["hist_edges"][:-1])
        for i in range(nb):
            for j in range(ne):
                a0, a1, a2 = axs[j, 3 * i], axs[j, 3 * i + 1], axs[j, 3 * i + 2]
                for k, tr in enumerate(d["traj"][i, j]):
                    a0.plot(x, tr, lw=0.6, color=plt.get_cmap("Blues")(0.3 + 0.7 * k / 16))
                a0.set_ylim(-1.05, 1.05)
                a1.plot(d["hist"][i, j], mid, "k-"); a1.set_xticks([]); a1.set_yticks([])
                a2.plot(d["alpha_tau"][::3], d["C"][i, j][::3], "o", mfc="none", ms=3, color="k")
                a2.plot(t["alpha_tau"], t["C19"][i, j], "-", color="gray")
                a2.set_ylim(-0.3, 1.05)
                if j == 0:
                    a0.set_title(fr"$\beta={d['betas'][i]}$")
                if j == ne - 1:
                    a0.set_xlabel(r"$\alpha(t-t_0)$"); a2.set_xlabel(r"$\alpha\tau$")
                a0.set_ylabel(fr"$\varepsilon={d['eps'][j]}$")
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig19_finite_beta")


# ------------------------------------------------------------------- benchmark figures --
class FigOptimizerScaling(FigureOutput):
    depends_on = (B.PlannerBenchmark,)
    paper_ref = "Planner study (Eq. 13, fn. 58)"
    description = "Per-solver success probability, residual and time-to-solution vs N; breakdown sizes."

    def compute(self, deps):
        d = deps["PlannerBenchmark"]
        names = [str(x) for x in d["solvers"]]
        Ns = d["Ns"]
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
        cm = plt.get_cmap("tab10")
        for s, n in enumerate(names):
            if n == "exact":
                continue
            c = cm(s % 10)
            axs[0].plot(Ns, d["p_median"][s], "o-", ms=3, color=c, label=n)
            axs[1].semilogy(Ns, np.maximum(d["residual_median"][s], 1e-6), "o-", ms=3, color=c)
            tts = np.where(np.isfinite(d["tts99"][s]), d["tts99"][s], np.nan)
            axs[2].semilogy(Ns, tts, "o-", ms=3, color=c)
        ex = Ns[d["ref_exact"].astype(bool)]
        for ax in axs:
            if len(ex):
                ax.axvspan(Ns.min(), ex.max(), color="#d1e7dd", alpha=0.4, lw=0)
            ax.set_xlabel("N")
        axs[0].set(ylabel=r"median $p_{\rm success}$ (vs reference)", ylim=(-0.05, 1.05))
        axs[0].legend(fontsize=6, ncol=2)
        axs[1].set(ylabel=r"median residual $(\mathcal{R}_{\rm ref}-\mathcal{R})/\mathcal{R}_{\rm ref}$ (floor $10^{-6}$)")
        axs[2].set(ylabel=r"TTS$_{99}$ [s]")
        fig.suptitle("green band: reference certified by exact enumeration; beyond it: best-known", fontsize=8)
        fig.tight_layout()
        return save_fig(fig, self.folder(), "fig_optimizer_scaling")


class FigSatisficingGap(FigureOutput):
    depends_on = (D.RewardScanN, B.PlannerBenchmark, T.AnalyticCurves)
    paper_ref = "Sec. III A, IX A"
    description = "Learning agents vs planner: R_N of SK-game learning vs best planner value vs Parisi, against N^{-2/3}."

    def compute(self, deps):
        g, b = deps["RewardScanN"], deps["PlannerBenchmark"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        i0 = list(g["eps"]).index(0.0) if 0.0 in list(g["eps"]) else 0
        x = g["Ns"] ** (-2 / 3)
        ax.errorbar(x, g["reward"][i0].mean(-1), ci95(g["reward"][i0], -1), fmt="o", mfc="none", label="SK-game learning (α=0.01)")
        xb = b["Ns"] ** (-2 / 3.0)
        ax.plot(xb, np.nanmean(b["ref"], -1), "s", color="k", mfc="none", label="planner: best known / exact")
        ub = np.nanmean(b["upper_bound"], -1)
        ax.plot(xb, ub, "^", color="gray", mfc="none", label="SDP upper bound")
        xx = np.linspace(0, max(x.max(), xb.max()), 20)
        ax.plot(xx, R_inf(0) - 1.5 * xx, "k--", lw=0.8, label=r"$\mathcal{R}_\infty-1.5N^{-2/3}$")
        ax.set(xlabel=r"$N^{-2/3}$", ylabel=r"$\mathcal{R}_N$", title=r"satisficing gap, $\varepsilon=0$")
        ax.legend(fontsize=6)
        return save_fig(fig, self.folder(), "fig_satisficing_gap")
