"""Direct-simulation data classes (finite N). One class per computation of the paper."""
from __future__ import annotations

import numpy as np

from ..registry import Output, stable_hash
from ..model import GameParams, SKGame, WindowRecorder, SnapshotRecorder, TwoTimeRecorder, INF, InteractionEnsemble
from ..observables import (stationary_autocorr, individual_rewards, average_reward, cross_overlap,
                           frozen_fraction, classify_phase, dominant_frequency, q_param, magnetisation)
from ..theory.analytic import Tc
from ..theory.complexity import count_fixed_points_T0, count_nmfe_solutions, annealed_complexity
from .common import simulate_window, beta_of


def _seed(obj, *k):
    """Deterministic per-(class, key) seed (independent of PYTHONHASHSEED)."""
    return int(obj.rng(int(stable_hash(list(map(str, k)))[:8], 16)).integers(2**31))


def _final_reward(N, alpha, eps, beta, R, t_eq, window, seed, J0=0.0):
    """Average reward over the last `window` steps, per realisation."""
    g, w = simulate_window(N, alpha, eps, beta, R, t_eq, window, seed=seed, J0=J0, store_e=True, store_S=True)
    e = w["e"].astype(float)            # (R, W, N) individual realised rewards S_i R_i
    return e.mean(axis=(1, 2)), g, w


class PhaseClassification(Output):
    paper_ref = "Fig. 1 (numerical counterpart)"
    description = "Classify (eps, alpha) at beta=inf, finite N: fixed point / L-cycle / no recurrence."
    presets = {"quick": {"N": 64, "R": 8, "alphas": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                         "eps": list(np.round(np.linspace(0, 2, 11), 2)), "t_alpha": 300, "window": 64},
               "paper": {"N": 128, "R": 32, "alphas": list(np.round(np.linspace(1, 0.05, 20), 3)),
                         "eps": list(np.round(np.linspace(0, 2, 41), 3)), "t_alpha": 3000, "window": 128}}

    def compute(self, deps):
        p = self.params
        frac = np.zeros((len(p["alphas"]), len(p["eps"]), 4))  # FP, L=2, L=4, other
        for i, a in enumerate(p["alphas"]):
            for j, e in enumerate(p["eps"]):
                t_eq = int(p["t_alpha"] / a)
                _, w = simulate_window(p["N"], a, e, INF, p["R"], t_eq, p["window"], seed=_seed(self, i, j))
                L = classify_phase(w["S"])
                frac[i, j] = [np.mean(L == 1), np.mean(L == 2), np.mean(L == 4), np.mean(~np.isin(L, [1, 2, 4]))]
        return {"alphas": np.array(p["alphas"]), "eps": np.array(p["eps"]), "frac": frac}


class RewardScanAlpha(Output):
    paper_ref = "Fig. 2(a)"
    description = "Average reward vs memory-loss rate alpha for several eps, beta=inf."
    presets = {"quick": {"N": 128, "R": 8, "alphas": list(np.round(np.linspace(0.6, 1.0, 9), 3)),
                         "eps": [0, 0.6, 0.85, 1.05, 1.5, 2.0], "t_eq": 2000, "window": 400},
               "paper": {"N": 256, "R": 32, "alphas": list(np.round(np.linspace(0.6, 1.0, 21), 3)),
                         "eps": [0, 0.6, 0.85, 1.05, 1.5, 2.0], "t_eq": 20000, "window": 2000}}

    def compute(self, deps):
        p = self.params
        Rm = np.zeros((len(p["eps"]), len(p["alphas"]), p["R"]))
        for i, e in enumerate(p["eps"]):
            for j, a in enumerate(p["alphas"]):
                Rm[i, j], _, _ = _final_reward(p["N"], a, e, INF, p["R"], p["t_eq"], p["window"], _seed(self, i, j))
        return {"eps": np.array(p["eps"]), "alphas": np.array(p["alphas"]), "reward": Rm, "N": p["N"]}


class RewardScanEps(Output):
    paper_ref = "Fig. 2(b)"
    description = "Average reward vs asymmetry eps for alpha in {0.01, 0.1}, beta=inf."
    presets = {"quick": {"N": 128, "R": 8, "alphas": [0.01, 0.1], "eps": list(np.round(np.linspace(0, 2, 11), 2)),
                         "t_alpha": 60, "window_alpha": 4},
               "paper": {"N": 256, "R": 32, "alphas": [0.01, 0.1], "eps": list(np.round(np.linspace(0, 2, 41), 3)),
                         "t_alpha": 1000, "window_alpha": 20}}

    def compute(self, deps):
        p = self.params
        Rm = np.zeros((len(p["alphas"]), len(p["eps"]), p["R"]))
        for i, a in enumerate(p["alphas"]):
            for j, e in enumerate(p["eps"]):
                Rm[i, j], _, _ = _final_reward(p["N"], a, e, INF, p["R"], int(p["t_alpha"] / a),
                                               int(p["window_alpha"] / a), _seed(self, i, j))
        return {"eps": np.array(p["eps"]), "alphas": np.array(p["alphas"]), "reward": Rm, "N": p["N"]}


class RewardScanNoise(Output):
    paper_ref = "Fig. 2(c)"
    description = "Average reward vs noise 1/beta (alpha=0.01) for eps in {0, 0.6}: non-monotonic."
    presets = {"quick": {"N": 128, "R": 8, "alpha": 0.01, "eps": [0.0, 0.6], "T": list(np.round(np.linspace(0, 1, 11), 2)),
                         "t_alpha": 60, "window_alpha": 4},
               "paper": {"N": 256, "R": 32, "alpha": 0.01, "eps": [0.0, 0.6], "T": list(np.round(np.linspace(0, 1, 26), 3)),
                         "t_alpha": 1000, "window_alpha": 20}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        Rm = np.zeros((len(p["eps"]), len(p["T"]), p["R"]))
        for i, e in enumerate(p["eps"]):
            for j, T in enumerate(p["T"]):
                Rm[i, j], _, _ = _final_reward(p["N"], a, e, beta_of(T), p["R"], int(p["t_alpha"] / a),
                                               int(p["window_alpha"] / a), _seed(self, i, j))
        return {"eps": np.array(p["eps"]), "T": np.array(p["T"]), "reward": Rm, "N": p["N"]}


class RewardScanN(Output):
    paper_ref = "Fig. 2(d)"
    description = "Average reward vs N (alpha=0.01, beta=inf), fit R_N = R_inf - A N^{-2/3}."
    presets = {"quick": {"Ns": [32, 64, 128, 256], "R": 8, "alpha": 0.01, "eps": [0.0, 0.6], "t_alpha": 60,
                         "window_alpha": 4},
               "paper": {"Ns": [32, 64, 128, 256, 512, 1024], "R": 32, "alpha": 0.01, "eps": [0.0, 0.6],
                         "t_alpha": 1000, "window_alpha": 20}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        Rm = np.zeros((len(p["eps"]), len(p["Ns"]), p["R"]))
        for i, e in enumerate(p["eps"]):
            for j, N in enumerate(p["Ns"]):
                Rm[i, j], _, _ = _final_reward(N, a, e, INF, p["R"], int(p["t_alpha"] / a), int(p["window_alpha"] / a),
                                               _seed(self, i, j))
        # fit R = Rinf - A N^-2/3 excluding the smallest N
        fits = []
        x = np.array(p["Ns"], float) ** (-2 / 3)
        for i in range(len(p["eps"])):
            y = Rm[i].mean(-1)
            sel = np.array(p["Ns"]) > min(p["Ns"])
            A_, R0 = np.polyfit(x[sel], y[sel], 1)
            fits.append((R0, -A_))
        return {"eps": np.array(p["eps"]), "Ns": np.array(p["Ns"]), "reward": Rm, "fit": np.array(fits)}


class AutocorrAlpha1(Output):
    paper_ref = "Figs. 3, 11"
    description = "Stationary C(tau) at alpha=1, beta=inf (Fig. 3) and C(tau=2) vs eps for several N (Fig. 11)."
    presets = {"quick": {"N3": 256, "R3": 16, "eps3": [0.1, 0.85, 1.05, 1.5], "t_eq": 3000, "window": 300,
                         "Ns11": [64, 128, 256], "R11": 8, "eps11": list(np.round(np.linspace(0.5, 1.0, 11), 3)),
                         "t_eq11": 4000},
               "paper": {"N3": 256, "R3": 128, "eps3": [0.1, 0.85, 1.05, 1.5], "t_eq": 100000, "window": 1000,
                         "Ns11": [64, 128, 256, 512, 1024], "R11": 128, "eps11": list(np.round(np.linspace(0.5, 1.0, 26), 3)),
                         "t_eq11": 5000000}}

    def compute(self, deps):
        p = self.params
        C3 = []
        for i, e in enumerate(p["eps3"]):
            _, w = simulate_window(p["N3"], 1.0, e, INF, p["R3"], p["t_eq"], p["window"], seed=_seed(self, 3, i))
            C3.append(stationary_autocorr(w["S"], 6))
        C11 = np.zeros((len(p["Ns11"]), len(p["eps11"]), p["R11"]))
        for i, N in enumerate(p["Ns11"]):
            for j, e in enumerate(p["eps11"]):
                _, w = simulate_window(N, 1.0, e, INF, p["R11"], p["t_eq11"], 100, seed=_seed(self, 11, i, j))
                C11[i, j] = stationary_autocorr(w["S"], 2)[:, 2]
        return {"eps3": np.array(p["eps3"]), "C3": np.array(C3), "Ns11": np.array(p["Ns11"]),
                "eps11": np.array(p["eps11"]), "C11": C11}


class RewardDistributions(Output):
    paper_ref = "Figs. 4, 6"
    description = "Distribution of individual rewards e_i (late times) for alpha in {1,0.5,0.01} and several eps."
    presets = {"quick": {"N": 256, "R": 6, "alphas": [1.0, 0.5, 0.01], "eps": [0, 0.1, 0.85, 1.05, 1.5, 2.0],
                         "t_alpha": 60, "window": 40, "N4": 256, "R4": 8},
               "paper": {"N": 256, "R": 32, "alphas": [1.0, 0.5, 0.01], "eps": [0, 0.1, 0.85, 1.05, 1.5, 2.0],
                         "t_alpha": 2000, "window": 200, "N4": 512, "R4": 128}}

    def compute(self, deps):
        p = self.params
        bins = np.linspace(-6, 6, 121)
        H = np.zeros((len(p["alphas"]), len(p["eps"]), len(bins) - 1))
        e4 = []
        for i, a in enumerate(p["alphas"]):
            for j, e in enumerate(p["eps"]):
                _, w = simulate_window(p["N"], a, e, INF, p["R"], int(p["t_alpha"] / a), p["window"],
                                       stride=max(1, p["window"] // 10), store_e=True, seed=_seed(self, i, j))
                H[i, j], _ = np.histogram(w["e"].ravel(), bins=bins, density=True)
            # Fig. 4: eps=0 at N4, final configuration only (fixed points)
            g, w = simulate_window(p["N4"], a, 0.0, INF, p["R4"], int(p["t_alpha"] / a), 1, store_e=True,
                                   seed=_seed(self, 4, i))
            e4.append(w["e"][:, -1].ravel())
        return {"alphas": np.array(p["alphas"]), "eps": np.array(p["eps"]), "bins": bins, "hist": H,
                "e_fig4": np.array(e4)}


class RewardTrajectories(Output):
    paper_ref = "Fig. 5"
    description = "Individual rewards e_i(t) on log-spaced times (alpha=0.01) for eps in {0, 0.6}."
    presets = {"quick": {"N": 256, "alpha": 0.01, "eps": [0.0, 0.6], "alpha_t_max": 100.0, "n_times": 60},
               "paper": {"N": 256, "alpha": 0.01, "eps": [0.0, 0.6], "alpha_t_max": 1000.0, "n_times": 120}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        times = np.unique(np.r_[0, np.geomspace(1, p["alpha_t_max"] / a, p["n_times"]).astype(int)])
        out = {"t": times, "alpha": a, "eps": np.array(p["eps"])}
        E = []
        for i, e in enumerate(p["eps"]):
            g = SKGame(GameParams(N=p["N"], alpha=a, eps=e), R=1, seed=_seed(self, i))
            rec = SnapshotRecorder(times)
            g.run(int(times[-1]) + 1, [rec])
            E.append(rec.result()["e"][0])     # (n_times, N)
        out["e"] = np.array(E)
        return out


class CrossOverlap(Output):
    paper_ref = "Fig. 7"
    description = "Overlap C^x_N of final individual rewards for two initial conditions, same J (eps=0)."
    presets = {"quick": {"Ns": [32, 64, 128, 256], "n_dis": 32, "alpha": 0.01, "t_alpha": 80},
               "paper": {"Ns": [32, 64, 128, 256, 512], "n_dis": 16 * 8, "alpha": 0.01, "t_alpha": 1000}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        Cx = []
        for i, N in enumerate(p["Ns"]):
            rng = self.rng(N)
            J = InteractionEnsemble(N, 0.0).sample(rng, p["n_dis"])
            J2 = np.concatenate([J, J], 0)                  # realisation r and r+n_dis share J
            g = SKGame(GameParams(N=N, alpha=a, eps=0.0), R=2 * p["n_dis"], seed=_seed(self, i), J=J2)
            g.run(int(p["t_alpha"] / a))
            S = g.intentions(g.Q)
            e = individual_rewards(S, g.J)
            Cx.append(cross_overlap(e[: p["n_dis"]], e[p["n_dis"]:]))
        return {"Ns": np.array(p["Ns"]), "Cx": np.array(Cx)}


class CooperationScan(Output):
    paper_ref = "Fig. 8"
    description = "Average reward and magnetisation vs cooperative bias J0 (alpha=0.01, beta=inf)."
    presets = {"quick": {"N": 128, "R": 4, "alpha": 0.01, "J0": list(np.round(np.linspace(0, 2, 9), 2)),
                         "eps": [0, 0.6, 0.85, 1.05, 1.5, 2.0], "t_alpha": 50, "window_alpha": 4},
               "paper": {"N": 256, "R": 32, "alpha": 0.01, "J0": list(np.round(np.linspace(0, 2, 21), 2)),
                         "eps": [0, 0.6, 0.85, 1.05, 1.5, 2.0], "t_alpha": 1000, "window_alpha": 20}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        Rm = np.zeros((len(p["eps"]), len(p["J0"]), p["R"]))
        M = np.zeros_like(Rm)
        for i, e in enumerate(p["eps"]):
            for j, J0 in enumerate(p["J0"]):
                Rm[i, j], g, w = _final_reward(p["N"], a, e, INF, p["R"], int(p["t_alpha"] / a),
                                               int(p["window_alpha"] / a), _seed(self, i, j), J0=J0)
                M[i, j] = np.abs(w["S"].astype(float).mean(axis=(1, 2)))
        return {"eps": np.array(p["eps"]), "J0": np.array(p["J0"]), "reward": Rm, "M": M}


class FixedPointCounts(Output):
    paper_ref = "Fig. 9"
    description = "Annealed complexity: exact T=0 enumeration vs eps; multistart NMFE counts on a (T, eps) grid."
    presets = {"quick": {"N_exact": 20, "n_samples": 24, "eps_exact": list(np.round(np.linspace(0, 1.2, 13), 2)),
                         "N_T": 14, "n_samples_T": 4, "T_over_2": [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
                         "eps_T": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25], "n_starts": 150},
               "paper": {"N_exact": 26, "n_samples": 100, "eps_exact": list(np.round(np.linspace(0, 1.2, 25), 3)),
                         "N_T": 40, "n_samples_T": 20, "T_over_2": list(np.round(np.linspace(0.025, 1, 20), 3)),
                         "eps_T": list(np.round(np.linspace(0, 1.25, 20), 3)), "n_starts": 4000}}

    def compute(self, deps):
        p = self.params
        rng = self.rng(0)
        N = p["N_exact"]
        Sig = []
        for e in p["eps_exact"]:
            Js = InteractionEnsemble(N, e).sample(rng, p["n_samples"])
            counts = [count_fixed_points_T0(J) for J in Js]
            Sig.append(annealed_complexity(counts, N))
        NT = p["N_T"]
        H = np.zeros((len(p["T_over_2"]), len(p["eps_T"])))
        for i, t2 in enumerate(p["T_over_2"]):
            for j, e in enumerate(p["eps_T"]):
                Js = InteractionEnsemble(NT, e).sample(rng, p["n_samples_T"])
                cnt = [count_nmfe_solutions(J, 1.0 / (2 * t2), p["n_starts"], rng)[0] for J in Js]
                H[i, j] = annealed_complexity(cnt, NT)
        return {"eps_exact": np.array(p["eps_exact"]), "Sigma_exact": np.array(Sig), "N_exact": N,
                "T_over_2": np.array(p["T_over_2"]), "eps_T": np.array(p["eps_T"]), "Sigma_T": H, "N_T": NT}


class FrozenVsAlpha(Output):
    paper_ref = "Fig. 12"
    description = "C(tau=1) vs alpha for eps<0.8 (memory freezes 2-cycles into fixed points) + sample trajectories."
    presets = {"quick": {"Ns": [64, 128], "R": 8, "alphas": list(np.round(np.linspace(0.5, 1.0, 11), 3)),
                         "eps": [0.0, 0.2, 0.4, 0.6, 0.79], "t_alpha": 400, "window": 50,
                         "traj_eps": 0.4, "traj_alphas": [1.0, 0.88, 0.7], "traj_N": 256},
               "paper": {"Ns": [128, 256], "R": 200, "alphas": list(np.round(np.linspace(0.5, 1.0, 26), 3)),
                         "eps": [0.0, 0.2, 0.4, 0.6, 0.79], "t_alpha": 1e5, "window": 100,
                         "traj_eps": 0.4, "traj_alphas": [1.0, 0.88, 0.7], "traj_N": 256}}

    def compute(self, deps):
        p = self.params
        C1 = np.zeros((len(p["Ns"]), len(p["eps"]), len(p["alphas"]), p["R"]))
        for i, N in enumerate(p["Ns"]):
            for j, e in enumerate(p["eps"]):
                for k, a in enumerate(p["alphas"]):
                    _, w = simulate_window(N, a, e, INF, p["R"], int(p["t_alpha"] / a), p["window"],
                                           seed=_seed(self, i, j, k))
                    C1[i, j, k] = stationary_autocorr(w["S"], 1)[:, 1]
        traj = []
        for k, a in enumerate(p["traj_alphas"]):
            _, w = simulate_window(p["traj_N"], a, p["traj_eps"], INF, 1, int(p["t_alpha"] / a), 50,
                                   seed=_seed(self, "traj", k))
            traj.append(w["S"][0][:, :32])
        return {"Ns": np.array(p["Ns"]), "eps": np.array(p["eps"]), "alphas": np.array(p["alphas"]), "C1": C1,
                "traj_alphas": np.array(p["traj_alphas"]), "traj": np.array(traj)}


class MemoryCorrelation(Output):
    paper_ref = "Fig. 13"
    description = "C(tau=2/alpha) vs eps for several alpha and N (finite-size vs DMFT) + trajectories at eps=1."
    presets = {"quick": {"Ns": [128, 256], "R": 8, "alphas": [1.0, 0.3, 0.1], "eps": [0.6, 0.7, 0.8, 0.9, 1.0],
                         "t_alpha": 300, "window_alpha": 10, "traj_alphas": [1.0, 0.1], "traj_N": 128},
               "paper": {"Ns": [128, 256, 512], "R": 32, "alphas": [1.0, 0.5, 0.2, 0.1, 0.05],
                         "eps": list(np.round(np.linspace(0.6, 1.0, 9), 3)), "t_alpha": 10000, "window_alpha": 40,
                         "traj_alphas": [1.0, 0.1], "traj_N": 128}}

    def compute(self, deps):
        p = self.params
        C = np.zeros((len(p["Ns"]), len(p["alphas"]), len(p["eps"]), p["R"]))
        for i, N in enumerate(p["Ns"]):
            for k, a in enumerate(p["alphas"]):
                lag = int(round(2 / a))
                for j, e in enumerate(p["eps"]):
                    _, w = simulate_window(N, a, e, INF, p["R"], int(p["t_alpha"] / a), int(p["window_alpha"] / a) + lag,
                                           seed=_seed(self, i, k, j))
                    C[i, k, j] = stationary_autocorr(w["S"], lag)[:, lag]
        traj = []
        for k, a in enumerate(p["traj_alphas"]):
            _, w = simulate_window(p["traj_N"], a, 1.0, INF, 1, int(p["t_alpha"] / a), int(40 / a),
                                   stride=max(1, int(0.1 / a)), seed=_seed(self, "t", k))
            traj.append(w["S"][0].T)   # (N, time); lengths differ between alphas
        return {"Ns": np.array(p["Ns"]), "alphas": np.array(p["alphas"]), "eps": np.array(p["eps"]), "C": C,
                "traj_alphas": np.array(p["traj_alphas"]), **{f"traj_{k}": t for k, t in enumerate(traj)}}


class Eps1Decorrelation(Output):
    paper_ref = "Fig. 14"
    description = "C(tau) at eps=1 for several alpha: collapse vs alpha*tau onto the Sompolinsky-Crisanti curve."
    presets = {"quick": {"Ns": [512], "R": 8, "alphas": [1.0, 0.3, 0.1, 0.03], "t_alpha": 60, "max_alpha_tau": 8},
               "paper": {"Ns": [512, 1024], "R": 20, "alphas": [1.0, 0.3, 0.1, 0.03, 0.01], "t_alpha": 500,
                         "max_alpha_tau": 8}}

    def compute(self, deps):
        p = self.params
        grid = np.linspace(0, p["max_alpha_tau"], 81)
        C = np.zeros((len(p["Ns"]), len(p["alphas"]), 81))
        raw = {}
        for i, N in enumerate(p["Ns"]):
            for k, a in enumerate(p["alphas"]):
                L = int(p["max_alpha_tau"] / a)
                _, w = simulate_window(N, a, 1.0, INF, p["R"], int(p["t_alpha"] / a), 3 * L, seed=_seed(self, i, k))
                c = stationary_autocorr(w["S"], L).mean(0)
                C[i, k] = np.interp(grid, a * np.arange(L + 1), c)
                raw[f"C_raw_{i}_{k}"] = c
        return {"Ns": np.array(p["Ns"]), "alphas": np.array(p["alphas"]), "alpha_tau": grid, "C": C, **raw}


class CyclePeriod(Output):
    paper_ref = "Fig. 15"
    description = "Quasi-cycle frequency omega_0 vs alpha for eps > 2 - eps_c (period ~ alpha^{-1/2})."
    presets = {"quick": {"N": 128, "R": 6, "eps": [1.2, 1.4, 1.6, 1.8, 2.0], "alphas": list(np.geomspace(1, 0.03, 7)),
                         "t_alpha": 40, "window": 1200, "eps_C": 1.8, "N_C": 256},
               "paper": {"N": 256, "R": 96, "eps": list(np.round(np.linspace(1.2, 2.0, 9), 2)),
                         "alphas": list(np.geomspace(1, 0.005, 15)), "t_alpha": 1000, "window": 8000,
                         "eps_C": 1.8, "N_C": 256}}

    def compute(self, deps):
        p = self.params
        w0 = np.zeros((len(p["eps"]), len(p["alphas"])))
        for i, e in enumerate(p["eps"]):
            for k, a in enumerate(p["alphas"]):
                _, w = simulate_window(p["N"], a, e, INF, p["R"], int(p["t_alpha"] / a), p["window"],
                                       seed=_seed(self, i, k))
                C = stationary_autocorr(w["S"], p["window"] // 2).mean(0)
                w0[i, k] = dominant_frequency(C)
        Cs, Ps = [], []
        grid = np.linspace(0, 6, 121)
        for k, a in enumerate(p["alphas"][-4:]):
            L = int(6 / np.sqrt(a)) + 2
            _, w = simulate_window(p["N_C"], a, p["eps_C"], INF, p["R"], int(p["t_alpha"] / a), 4 * L + p["window"],
                                   seed=_seed(self, "C", k))
            c = stationary_autocorr(w["S"], max(L, p["window"] // 2)).mean(0)
            Cs.append(np.interp(grid, np.sqrt(a) * np.arange(len(c)), c))
            P = np.abs(np.fft.rfft(c - c.mean(), n=4 * len(c))) ** 2 / len(c)
            f = np.fft.rfftfreq(4 * len(c))
            Ps.append(np.interp(np.linspace(0, 1, 200), f / np.sqrt(a), a * P, right=np.nan))
        return {"eps": np.array(p["eps"]), "alphas": np.array(p["alphas"]), "omega0": w0, "sqrt_alpha_tau": grid,
                "C_eps": np.array(Cs), "alphas_C": np.array(p["alphas"][-4:]), "w_over_sqrt_alpha": np.linspace(0, 1, 200),
                "power": np.array(Ps)}


class QHeatmap(Output):
    paper_ref = "Fig. 16"
    description = "q = C(0) on a (T, eps) grid for alpha in {0.5, 0.1, 0.01}."
    presets = {"quick": {"N": 128, "R": 2, "alphas": [0.5, 0.1, 0.01], "T_over_2": list(np.round(np.linspace(0.05, 1, 8), 3)),
                         "eps": list(np.round(np.linspace(0, 2, 9), 2)), "t_alpha": 30, "window_alpha": 3},
               "paper": {"N": 256, "R": 32, "alphas": [0.5, 0.1, 0.01], "T_over_2": list(np.round(np.linspace(0.02, 1, 30), 3)),
                         "eps": list(np.round(np.linspace(0, 2, 30), 3)), "t_alpha": 10000, "window_alpha": 20}}

    def compute(self, deps):
        p = self.params
        Q = np.zeros((len(p["alphas"]), len(p["T_over_2"]), len(p["eps"])))
        for k, a in enumerate(p["alphas"]):
            for i, t2 in enumerate(p["T_over_2"]):
                for j, e in enumerate(p["eps"]):
                    _, w = simulate_window(p["N"], a, e, 1.0 / (2 * t2), p["R"], int(p["t_alpha"] / a),
                                           max(10, int(p["window_alpha"] / a)), stride=max(1, int(0.5 / a)),
                                           store_m=True, store_S=False, seed=_seed(self, k, i, j))
                    Q[k, i, j] = float(np.mean(w["m"].astype(float) ** 2))
        return {"alphas": np.array(p["alphas"]), "T_over_2": np.array(p["T_over_2"]), "eps": np.array(p["eps"]), "q": Q}


class QvsTemperature(Output):
    paper_ref = "Fig. 17(a)"
    description = "Stationary q vs T/T_c(eps) at eps=0.1 for alpha in {0.01, 0.1} (finite N)."
    presets = {"quick": {"N": 256, "R": 4, "eps": 0.1, "alphas": [0.01, 0.1], "x_T": list(np.round(np.linspace(0.05, 1.2, 12), 3)),
                         "t_alpha": 40, "window_alpha": 4},
               "paper": {"N": 256, "R": 32, "eps": 0.1, "alphas": [0.01, 0.1], "x_T": list(np.round(np.linspace(0.02, 1.2, 40), 3)),
                         "t_alpha": 10000, "window_alpha": 40}}

    def compute(self, deps):
        p = self.params
        Tc_ = Tc(p["eps"])
        q = np.zeros((len(p["alphas"]), len(p["x_T"])))
        for k, a in enumerate(p["alphas"]):
            for i, x in enumerate(p["x_T"]):
                _, w = simulate_window(p["N"], a, p["eps"], 1.0 / (x * Tc_), p["R"], int(p["t_alpha"] / a),
                                       int(p["window_alpha"] / a), stride=max(1, int(0.5 / a)), store_m=True,
                                       store_S=False, seed=_seed(self, k, i))
                q[k, i] = float(np.mean(w["m"].astype(float) ** 2))
        return {"alphas": np.array(p["alphas"]), "x_T": np.array(p["x_T"]), "q": q, "eps": p["eps"]}


class Aging(Output):
    paper_ref = "Fig. 18"
    description = "Two-time correlation C(t_w, t_w+t): t/t_w scaling (eps<eps*) vs TTI (eps>eps*)."
    presets = {"quick": {"N": 256, "R": 16, "sets": [[0.5, 4.0, 0.0], [0.2, 2.0, 0.25], [0.5, 4.0, 0.5], [0.2, 2.0, 0.5]],
                         "alpha_tw": [10, 30, 100, 300, 1000], "max_lag_factor": 10, "n_lags": 40},
               "paper": {"N": 256, "R": 576, "sets": [[0.5, 4.0, 0.0], [0.2, 2.0, 0.25], [0.5, 4.0, 0.5], [0.2, 2.0, 0.5]],
                         "alpha_tw": [1e3, 1e4, 1e5, 1e6, 1e7], "max_lag_factor": 100, "n_lags": 60}}

    def compute(self, deps):
        p = self.params
        out = {"sets": np.array(p["sets"]), "alpha_tw": np.array(p["alpha_tw"], float)}
        for s, (a, b, e) in enumerate(p["sets"]):
            tws = [int(x / a) for x in p["alpha_tw"]]
            lags = np.unique(np.geomspace(1, p["max_lag_factor"] * max(tws), p["n_lags"]).astype(int))
            rec = TwoTimeRecorder(tws, lags)
            g = SKGame(GameParams(N=p["N"], alpha=a, beta=b, eps=e), R=p["R"], seed=_seed(self, s))
            g.run(max(tws) + int(lags.max()) + 1, [rec])
            r = rec.result()
            out[f"C_{s}"] = np.nanmean(r["C"], axis=-1)
            out[f"lags_{s}"] = lags
            out[f"tw_{s}"] = np.array(tws)
        return out


class FiniteBetaTrajectories(Output):
    paper_ref = "Fig. 19"
    description = "Intentions m_i(t), their histogram and C(tau) at alpha=0.1 for beta in {4,2} and four eps."
    presets = {"quick": {"N": 256, "R": 6, "alpha": 0.1, "betas": [4.0, 2.0], "eps": [0.1, 0.85, 1.05, 1.5],
                         "t0": 3000, "window_alpha": 30},
               "paper": {"N": 256, "R": 96, "alpha": 0.1, "betas": [4.0, 2.0], "eps": [0.1, 0.85, 1.05, 1.5],
                         "t0": 100000000, "window_alpha": 200}}

    def compute(self, deps):
        p = self.params
        a = p["alpha"]
        nb, ne = len(p["betas"]), len(p["eps"])
        grid = np.linspace(0, 8, 81)
        W = int(p["window_alpha"] / a)
        traj = np.zeros((nb, ne, 16, int(8 / a)))
        hist = np.zeros((nb, ne, 40))
        C = np.zeros((nb, ne, 81))
        for i, b in enumerate(p["betas"]):
            for j, e in enumerate(p["eps"]):
                _, w = simulate_window(p["N"], a, e, b, p["R"], p["t0"], W, store_m=True, store_S=False,
                                       seed=_seed(self, i, j))
                m = w["m"]
                traj[i, j] = m[0, : int(8 / a), :16].T
                hist[i, j], _ = np.histogram(m.ravel(), bins=np.linspace(-1, 1, 41), density=True)
                L = int(8 / a)
                c = stationary_autocorr(m, L).mean(0)
                C[i, j] = np.interp(grid, a * np.arange(L + 1), c)
        return {"betas": np.array(p["betas"]), "eps": np.array(p["eps"]), "traj": traj, "hist": hist,
                "hist_edges": np.linspace(-1, 1, 41), "alpha_tau": grid, "C": C, "alpha": a}
