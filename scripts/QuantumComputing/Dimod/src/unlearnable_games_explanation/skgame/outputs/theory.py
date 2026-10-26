"""Theory output classes (analytic, symbolic, DMFT, complexity)."""
from __future__ import annotations

import json

import numpy as np

from ..registry import Output
from ..model import upsilon, eta
from ..theory import analytic as A
from ..theory import symbolic
from ..theory.complexity import Sigma2_curve
from ..theory.dmft import DMFTSolver


class SymbolicChecks(Output):
    kind = "table"
    paper_ref = "Eqs. 9-10, 13, 18, 20, 22-23, 33, 37-39, 49-50, B37, B50"
    description = "SymPy verification of every closed-form step (incl. two corrected typos)."

    def compute(self, deps):
        rows = symbolic.run_all()
        folder = self.folder()
        md = ["| check | paper | claim | passed | detail |", "|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['name']} | {r['eq']} | {r['claim']} | {'✅' if r['passed'] else '❌'} | "
                      f"{r['detail'][:160].replace('|', '/')} |")
        (folder / "checks.md").write_text("\n".join(md) + "\n")
        (folder / "checks.json").write_text(json.dumps(rows, indent=2))
        return {"n_checks": len(rows), "n_passed": int(sum(r["passed"] for r in rows))}


class AnalyticCurves(Output):
    kind = "theory"
    paper_ref = "Eqs. 9, 10, 13, 20, 22, 37-39"
    description = "ups(eps), eta(eps), R_inf(eps), T_c(eps), Sigma_FP(eps), Sompolinsky-Crisanti C(tau)."
    presets = {"quick": {"n_eps": 201}, "paper": {"n_eps": 401}}

    def compute(self, deps):
        eps = np.linspace(0, 2, self.params["n_eps"])
        tau, Csc, Dsc = A.sompolinsky_crisanti()
        eps_c = eps[eps <= 1.0]
        return {"eps": eps, "upsilon": upsilon(eps), "eta": eta(eps), "R_inf": A.R_inf(eps),
                "Tc": A.Tc(eps), "eps_le1": eps_c, "Sigma_fp": A.complexity_fp_T0_eps(eps_c),
                "Tc_stability": np.array([A.Tc_from_stability(e) for e in eps[(eps < 1.999)][::10]]),
                "eps_stab": eps[(eps < 1.999)][::10],
                "sc_tau": tau, "sc_C": Csc, "sc_Delta": Dsc, "tau1": A.TAU1, "Delta0": A.DELTA0}


class StaticDMFTSolution(Output):
    kind = "theory"
    paper_ref = "Eqs. 40-50, Fig. 17"
    description = "RS/static fixed point (q, chi) of the DMFT vs T/Tc and spectral density of perturbations."
    presets = {"quick": {"eps": 0.1, "n_T": 40}, "paper": {"eps": 0.1, "n_T": 100}}

    def compute(self, deps):
        e = self.params["eps"]
        Tc = A.Tc(e)
        x = np.linspace(0.02, 1.15, self.params["n_T"])
        qc = A.static_dmft_curve(1.0 / (x * Tc), e)
        xs = np.linspace(0.8, 1.2, 81)
        spec = np.array([A.inverse_spectral_density_omega0(1.0 / (xx * Tc), e) for xx in xs])
        return {"T_over_Tc": x, "q": qc[:, 0], "chi": qc[:, 1], "x_spec": xs, "inv_spec": spec,
                "eps": e, "Tc": float(Tc)}


class CycleComplexity(Output):
    kind = "theory"
    paper_ref = "Eq. 25, Appendix B, Fig. 10"
    description = "Annealed complexity Sigma_2(alpha, eps) of 2-cycles and 1-C(1) at the saddle."
    presets = {"quick": {"eps": [0.0, 0.2, 0.4, 0.6], "n_alpha": 120, "alpha_min": 0.02},
               "paper": {"eps": [0.0, 0.2, 0.4, 0.6], "n_alpha": 400, "alpha_min": 0.005}}

    def compute(self, deps):
        al = np.linspace(1.0, self.params["alpha_min"], self.params["n_alpha"])
        out = {"alpha": al, "eps": np.array(self.params["eps"])}
        S2, oneC, conv, Sfp = [], [], [], []
        for e in self.params["eps"]:
            c = Sigma2_curve(al, eta(e))
            S2.append(c[:, 1]); oneC.append(c[:, 2]); conv.append(c[:, 3])
            Sfp.append(A.complexity_fp_T0(eta(e))[0])
        out.update(Sigma2=np.array(S2), one_minus_C1=np.array(oneC), converged=np.array(conv),
                   Sigma_fp=np.array(Sfp))
        return out


class DMFTAlpha1(Output):
    kind = "theory"
    paper_ref = "Eq. 30, Figs. 3, 11"
    description = "Discrete DMFT at alpha=1, beta=inf: C(tau) for Fig. 3 and C(tau=2) vs eps for Fig. 11."
    presets = {"quick": {"eps_fig3": [0.1, 0.85, 1.05, 1.5], "eps_fig11": list(np.round(np.linspace(0.5, 1.0, 11), 3)),
                         "T": 60, "M": 12000, "n_iter": 20, "lags": 6},
               "paper": {"eps_fig3": [0.1, 0.85, 1.05, 1.5], "eps_fig11": list(np.round(np.linspace(0.5, 1.0, 26), 3)),
                         "T": 200, "M": 50000, "n_iter": 40, "lags": 6}}

    def compute(self, deps):
        p = self.params
        C3 = []
        for e in p["eps_fig3"]:
            r = DMFTSolver(1.0, np.inf, e, T=p["T"], M=p["M"], n_iter=p["n_iter"], seed=1).solve()
            C3.append(r.stationary_C(p["lags"], p["T"] // 3))
        C11 = []
        for e in p["eps_fig11"]:
            r = DMFTSolver(1.0, np.inf, e, T=p["T"], M=p["M"], n_iter=p["n_iter"], seed=2).solve()
            C11.append(r.stationary_C(2, p["T"] // 3)[2])
        return {"eps_fig3": np.array(p["eps_fig3"]), "C_fig3": np.array(C3),
                "eps_fig11": np.array(p["eps_fig11"]), "C2_fig11": np.array(C11)}


class DMFTMemory(Output):
    kind = "theory"
    paper_ref = "Eq. 30, Figs. 13, 14, 15"
    description = "Discrete DMFT with memory (alpha<1, beta=inf): C(2/alpha) vs eps, C(tau) at eps=1, cycle frequency."
    presets = {"quick": {"alphas": [1.0, 0.3, 0.1], "eps": [0.6, 0.7, 0.8, 0.9, 1.0], "T_per_alpha": 14,
                         "M": 6000, "n_iter": 14, "alphas_eps1": [1.0, 0.3, 0.1],
                         "eps_cycles": 1.5, "alphas_cycles": [0.5, 0.2, 0.1]},
               "paper": {"alphas": [1.0, 0.5, 0.2, 0.1, 0.05], "eps": list(np.round(np.linspace(0.6, 1.0, 9), 3)),
                         "T_per_alpha": 30, "M": 30000, "n_iter": 30, "alphas_eps1": [1.0, 0.5, 0.2, 0.1, 0.05],
                         "eps_cycles": 1.5, "alphas_cycles": [0.5, 0.2, 0.1, 0.05, 0.02]}}

    def _solve(self, alpha, eps, seed):
        p = self.params
        T = int(min(400, max(40, p["T_per_alpha"] / alpha)))
        return DMFTSolver(alpha, np.inf, eps, T=T, M=p["M"], n_iter=p["n_iter"], seed=seed).solve(), T

    def compute(self, deps):
        p = self.params
        C2a = np.zeros((len(p["alphas"]), len(p["eps"])))
        for i, a in enumerate(p["alphas"]):
            lag = int(round(2 / a))
            for j, e in enumerate(p["eps"]):
                r, T = self._solve(a, e, 10 * i + j)
                C2a[i, j] = r.stationary_C(lag, T // 3 + lag)[lag]
        Ceps1 = []
        for a in p["alphas_eps1"]:
            r, T = self._solve(a, 1.0, 99)
            lags = np.arange(0, T // 2)
            C = r.stationary_C(len(lags) - 1, T // 2 + 1)
            Ceps1.append(np.interp(np.linspace(0, 8, 81), a * lags, C, right=np.nan))
        w0 = []
        for a in p["alphas_cycles"]:
            r, T = self._solve(a, p["eps_cycles"], 7)
            C = r.stationary_C(T // 2, T // 2 + 1)
            from ..observables import dominant_frequency
            w0.append(dominant_frequency(C))
        return {"alphas": np.array(p["alphas"]), "eps": np.array(p["eps"]), "C_2_over_alpha": C2a,
                "alphas_eps1": np.array(p["alphas_eps1"]), "alpha_tau": np.linspace(0, 8, 81),
                "C_eps1": np.array(Ceps1), "alphas_cycles": np.array(p["alphas_cycles"]),
                "omega0": np.array(w0), "eps_cycles": p["eps_cycles"]}


class DMFTFiniteBeta(Output):
    kind = "theory"
    paper_ref = "Eq. 30-31, Figs. 17a, 19"
    description = "Discrete DMFT at finite beta (online noise): q vs T/Tc and C(tau) for Fig. 19."
    presets = {"quick": {"alpha": 0.1, "eps_q": 0.1, "x_T": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], "T": 160, "M": 4000,
                         "n_iter": 10, "betas_19": [4.0, 2.0], "eps_19": [0.1, 0.85, 1.05, 1.5]},
               "paper": {"alpha": 0.1, "eps_q": 0.1, "x_T": list(np.round(np.linspace(0.05, 1.2, 20), 3)),
                         "T": 400, "M": 20000, "n_iter": 25, "betas_19": [4.0, 2.0], "eps_19": [0.1, 0.85, 1.05, 1.5]}}

    def compute(self, deps):
        p = self.params
        a, T = p["alpha"], p["T"]
        Tc = A.Tc(p["eps_q"])
        q = []
        for x in p["x_T"]:
            r = DMFTSolver(a, 1.0 / (x * Tc), p["eps_q"], T=T, M=p["M"], n_iter=p["n_iter"], seed=3).solve()
            q.append(np.mean(r.q[-T // 3:]))
        C19 = np.zeros((len(p["betas_19"]), len(p["eps_19"]), 81))
        for i, b in enumerate(p["betas_19"]):
            for j, e in enumerate(p["eps_19"]):
                r = DMFTSolver(a, b, e, T=T, M=p["M"], n_iter=p["n_iter"], seed=4).solve()
                lags = np.arange(0, T // 2)
                C = r.stationary_C(len(lags) - 1, T // 2 + 1)
                C19[i, j] = np.interp(np.linspace(0, 8, 81), a * lags, C, right=np.nan)
        return {"x_T": np.array(p["x_T"]), "q": np.array(q), "alpha_tau": np.linspace(0, 8, 81), "C19": C19,
                "betas_19": np.array(p["betas_19"]), "eps_19": np.array(p["eps_19"])}
