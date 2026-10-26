"""The *mathematical* category of the paper: objects = mathematical objects, generators =
computations between them. Every output class is mapped to the computations it uses
(OUTPUT_USES), which defines a relation  Outputs -> Morphisms  exported for Catlab.

This is the machine-readable form of docs/02_objects.md and docs/03_computations.md.
"""
from __future__ import annotations

# name: (symbol, space / type, paper reference, python implementation)
OBJECTS = {
    "Params":        ("(N,α,β,ε,J0,Jd)", "parameter space", "Sec. II", "model.GameParams"),
    "Ensemble":      ("(υ,η)", "moments of the J law", "Eqs. 9-10", "model.upsilon / model.eta"),
    "J":             ("J_ij", "R^{N×N}", "Eqs. 8, 11", "model.InteractionEnsemble.sample"),
    "Q":             ("Q_i(t)", "R^N", "Eq. 3", "model.SKGame.Q"),
    "m":             ("m_i(t)", "[-1,1]^N", "Eq. 2", "model.SKGame.intentions"),
    "S":             ("S_i(t)", "{-1,1}^N", "Eq. 1", "model.SKGame.decisions"),
    "Rvec":          ("R_i(t)", "R^N", "Eq. 4", "model.SKGame.rewards"),
    "Trajectory":    ("{Q,m,S}_{t}", "(R^N)^T", "Eqs. 5, 14", "model.SKGame.run + recorders"),
    "RN":            ("R_N", "R", "Eq. 12", "observables.average_reward"),
    "e":             ("e_i = S*_i R*_i", "R^N", "Eq. 16", "observables.individual_rewards"),
    "Corr":          ("C(t,t+τ), C(τ)", "R^{T×T}", "Eq. 15", "observables.stationary_autocorr"),
    "q":             ("q = C(t,t)", "[0,1]", "Sec. IV", "observables.q_param"),
    "M":             ("M(t)", "[-1,1]", "Eq. 11", "observables.magnetisation"),
    "Overlap":       ("C×_N", "R", "Eq. 17", "observables.cross_overlap"),
    "Phase":         ("FP / L-cycle / chaos", "label", "Fig. 1", "observables.classify_phase"),
    "Spectrum":      ("λ_max(J)", "R", "Eq. 20 (elliptic law)", "theory.analytic.lambda_max_elliptic"),
    "Tc":            ("T_c(ε)", "R_+", "Eq. 20", "theory.analytic.Tc"),
    "FixedPoints":   ("{m*: m*=tanh(βJm*)}", "subset of [-1,1]^N", "Eq. 19 (NMFE)", "theory.complexity.count_*"),
    "Sigma":         ("Σ(β,ε)", "R", "Eqs. 21-23", "theory.analytic.complexity_fp_T0"),
    "SigmaL":        ("Σ_L(α,η)", "R", "Eq. 25, B30", "theory.complexity.Sigma2_saddle"),
    "DMFT":          ("(C,G,q) effective process", "R^{T×T}×R^{T×T}", "Eqs. 30-35", "theory.dmft.DMFTSolver"),
    "SCSolution":    ("Δ(τ), τ1", "function", "Eqs. 36-39", "theory.analytic.sompolinsky_crisanti"),
    "RSStatic":      ("(q, χ)", "R^2", "Eqs. 40-44", "theory.analytic.static_dmft_fixed_point"),
    "Stability":     ("<|Q̂1(0)|²>^{-1}", "R", "Eqs. 45-50", "theory.analytic.inverse_spectral_density_omega0"),
    "Aging":         ("C(t_w, t_w+t)", "R^{n_w×n_t}", "Eqs. 51-52", "model.TwoTimeRecorder"),
    "Frequency":     ("ω0", "R_+", "Sec. VII C", "observables.dominant_frequency"),
    "Planner":       ("W = (J+Jᵀ)/2, H(S)", "quadratic form", "Eq. 12-13", "optim.problem.PlannerProblem"),
    "GroundState":   ("S* = argmax R_N", "{-1,1}^N", "Eq. 13", "optim.solvers.*"),
    "Parisi":        ("R_∞ = 0.7632(2-ε)", "R", "Eq. 13", "theory.analytic.R_inf"),
    "SymbolicProof": ("identities", "SymPy", "all closed forms", "theory.symbolic"),
}

# name: (domain objects, codomain, description)
MORPHISMS = {
    "moments":        (["Params"], "Ensemble", "υ = 1-ε+ε²/2, η = (1-ε)/υ"),
    "sample_J":       (["Params"], "J", "J = (1-ε/2)J^S + (ε/2)J^A + J0/N + Jd·1"),
    "intention":      (["Q", "Params"], "m", "m = tanh(βQ)"),
    "decide":         (["m"], "S", "logit sampling P(S=±1) = (1±m)/2"),
    "reward":         (["J", "S"], "Rvec", "R = J S"),
    "learn":          (["Q", "Rvec", "Params"], "Q", "Q ← (1-α)Q + αR"),
    "simulate":       (["J", "Params"], "Trajectory", "iterate intention∘decide∘reward∘learn"),
    "avg_reward":     (["J", "Trajectory"], "RN", "R_N = SᵀJS/N"),
    "indiv_reward":   (["J", "Trajectory"], "e", "e_i = S_i (JS)_i"),
    "autocorr":       (["Trajectory"], "Corr", "C(τ) = <m(t)·m(t+τ)>/N"),
    "order_param":    (["Trajectory"], "q", "q = <m²>"),
    "magnetise":      (["Trajectory"], "M", "M = ΣS/N"),
    "overlap":        (["e", "e"], "Overlap", "C× between two runs"),
    "classify":       (["Corr"], "Phase", "smallest L with C(nL)=1"),
    "two_time":       (["Trajectory"], "Aging", "C(t_w,t_w+t)"),
    "frequency":      (["Corr"], "Frequency", "peak of |FFT C|²"),
    "elliptic_edge":  (["Ensemble"], "Spectrum", "λ_max = √υ(1+η)"),
    "critical_T":     (["Spectrum"], "Tc", "β_c λ_max = 1"),
    "enumerate_fp":   (["J", "Params"], "FixedPoints", "Gray-code / multistart Newton"),
    "count":          (["FixedPoints"], "Sigma", "Σ = log E[N_J]/N"),
    "annealed_T0":    (["Ensemble"], "Sigma", "Eq. 22-23"),
    "cycle_saddle":   (["Ensemble", "Params"], "SigmaL", "saddle of Eq. B40"),
    "dmft_solve":     (["Ensemble", "Params"], "DMFT", "Monte-Carlo self-consistency + Novikov response"),
    "dmft_corr":      (["DMFT"], "Corr", "C(t,s) of the effective process"),
    "sc_limit":       (["DMFT"], "SCSolution", "ε=1, α→0 reduction (Eq. 36)"),
    "rs_static":      (["Ensemble", "Params"], "RSStatic", "Eqs. 42-44"),
    "perturb":        (["RSStatic", "Ensemble"], "Stability", "Eq. 48 at ω=0"),
    "stability_Tc":   (["Stability"], "Tc", "zero of Eq. 49 (corrected) = Eq. 20"),
    "planner":        (["J"], "Planner", "symmetrise, drop diagonal"),
    "optimise":       (["Planner", "Params"], "GroundState", "exact / heuristic / annealing"),
    "evaluate":       (["Planner", "GroundState"], "RN", "R_N(S*)"),
    "parisi":         (["Ensemble"], "Parisi", "R_∞ = 2·0.76317·(1-ε/2)"),
    "verify":         (["Ensemble"], "SymbolicProof", "SymPy identities"),
}

_SIM = ["sample_J", "intention", "decide", "reward", "learn", "simulate"]
OUTPUT_USES = {
    "SymbolicChecks": ["verify", "moments"],
    "AnalyticCurves": ["moments", "elliptic_edge", "critical_T", "annealed_T0", "parisi", "sc_limit"],
    "StaticDMFTSolution": ["moments", "rs_static", "perturb", "stability_Tc"],
    "CycleComplexity": ["moments", "cycle_saddle", "annealed_T0"],
    "DMFTAlpha1": ["moments", "dmft_solve", "dmft_corr"],
    "DMFTMemory": ["moments", "dmft_solve", "dmft_corr", "frequency"],
    "DMFTFiniteBeta": ["moments", "dmft_solve", "dmft_corr"],
    "PhaseClassification": _SIM + ["autocorr", "classify"],
    "RewardScanAlpha": _SIM + ["indiv_reward", "avg_reward"],
    "RewardScanEps": _SIM + ["indiv_reward", "avg_reward"],
    "RewardScanNoise": _SIM + ["indiv_reward", "avg_reward"],
    "RewardScanN": _SIM + ["indiv_reward", "avg_reward"],
    "AutocorrAlpha1": _SIM + ["autocorr"],
    "RewardDistributions": _SIM + ["indiv_reward"],
    "RewardTrajectories": _SIM + ["indiv_reward"],
    "CrossOverlap": _SIM + ["indiv_reward", "overlap"],
    "CooperationScan": _SIM + ["avg_reward", "magnetise"],
    "FixedPointCounts": ["sample_J", "enumerate_fp", "count"],
    "FrozenVsAlpha": _SIM + ["autocorr"],
    "MemoryCorrelation": _SIM + ["autocorr"],
    "Eps1Decorrelation": _SIM + ["autocorr"],
    "CyclePeriod": _SIM + ["autocorr", "frequency"],
    "QHeatmap": _SIM + ["order_param"],
    "QvsTemperature": _SIM + ["order_param", "critical_T"],
    "Aging": _SIM + ["two_time"],
    "FiniteBetaTrajectories": _SIM + ["autocorr"],
    "PlannerBenchmark": ["sample_J", "planner", "optimise", "evaluate"],
}


def check_well_typed():
    for n, (dom, cod, _) in MORPHISMS.items():
        for o in dom + [cod]:
            if o not in OBJECTS:
                raise ValueError(f"morphism {n}: unknown object {o}")
    for out, ms in OUTPUT_USES.items():
        for m in ms:
            if m not in MORPHISMS:
                raise ValueError(f"{out}: unknown morphism {m}")
    return True


def math_category_dict() -> dict:
    check_well_typed()
    return {"name": "SKGameMath",
            "objects": [{"name": k, "symbol": v[0], "space": v[1], "paper_ref": v[2], "impl": v[3]}
                        for k, v in OBJECTS.items()],
            "generators": [{"name": k, "dom": v[0], "codom": v[1], "description": v[2]}
                           for k, v in MORPHISMS.items()],
            "uses": [{"output": o, "morphism": m} for o, ms in OUTPUT_USES.items() for m in ms]}
