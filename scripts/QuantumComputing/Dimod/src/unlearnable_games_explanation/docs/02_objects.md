# 2. Mathematical objects

Generated from `skgame/mathcat.py` by `python -m skgame.docgen` — edit that file, not this one.

| object | symbol | space | paper | implementation |
|---|---|---|---|---|
| `Params` | (N,α,β,ε,J0,Jd) | parameter space | Sec. II | `skgame.model.GameParams` |
| `Ensemble` | (υ,η) | moments of the J law | Eqs. 9-10 | `skgame.model.upsilon / model.eta` |
| `J` | J_ij | R^{N×N} | Eqs. 8, 11 | `skgame.model.InteractionEnsemble.sample` |
| `Q` | Q_i(t) | R^N | Eq. 3 | `skgame.model.SKGame.Q` |
| `m` | m_i(t) | [-1,1]^N | Eq. 2 | `skgame.model.SKGame.intentions` |
| `S` | S_i(t) | {-1,1}^N | Eq. 1 | `skgame.model.SKGame.decisions` |
| `Rvec` | R_i(t) | R^N | Eq. 4 | `skgame.model.SKGame.rewards` |
| `Trajectory` | {Q,m,S}_{t} | (R^N)^T | Eqs. 5, 14 | `skgame.model.SKGame.run + recorders` |
| `RN` | R_N | R | Eq. 12 | `skgame.observables.average_reward` |
| `e` | e_i = S*_i R*_i | R^N | Eq. 16 | `skgame.observables.individual_rewards` |
| `Corr` | C(t,t+τ), C(τ) | R^{T×T} | Eq. 15 | `skgame.observables.stationary_autocorr` |
| `q` | q = C(t,t) | [0,1] | Sec. IV | `skgame.observables.q_param` |
| `M` | M(t) | [-1,1] | Eq. 11 | `skgame.observables.magnetisation` |
| `Overlap` | C×_N | R | Eq. 17 | `skgame.observables.cross_overlap` |
| `Phase` | FP / L-cycle / chaos | label | Fig. 1 | `skgame.observables.classify_phase` |
| `Spectrum` | λ_max(J) | R | Eq. 20 (elliptic law) | `skgame.theory.analytic.lambda_max_elliptic` |
| `Tc` | T_c(ε) | R_+ | Eq. 20 | `skgame.theory.analytic.Tc` |
| `FixedPoints` | {m*: m*=tanh(βJm*)} | subset of [-1,1]^N | Eq. 19 (NMFE) | `skgame.theory.complexity.count_*` |
| `Sigma` | Σ(β,ε) | R | Eqs. 21-23 | `skgame.theory.analytic.complexity_fp_T0` |
| `SigmaL` | Σ_L(α,η) | R | Eq. 25, B30 | `skgame.theory.complexity.Sigma2_saddle` |
| `DMFT` | (C,G,q) effective process | R^{T×T}×R^{T×T} | Eqs. 30-35 | `skgame.theory.dmft.DMFTSolver` |
| `SCSolution` | Δ(τ), τ1 | function | Eqs. 36-39 | `skgame.theory.analytic.sompolinsky_crisanti` |
| `RSStatic` | (q, χ) | R^2 | Eqs. 40-44 | `skgame.theory.analytic.static_dmft_fixed_point` |
| `Stability` | <|Q̂1(0)|²>^{-1} | R | Eqs. 45-50 | `skgame.theory.analytic.inverse_spectral_density_omega0` |
| `Aging` | C(t_w, t_w+t) | R^{n_w×n_t} | Eqs. 51-52 | `skgame.model.TwoTimeRecorder` |
| `Frequency` | ω0 | R_+ | Sec. VII C | `skgame.observables.dominant_frequency` |
| `Planner` | W = (J+Jᵀ)/2, H(S) | quadratic form | Eq. 12-13 | `skgame.optim.problem.PlannerProblem` |
| `GroundState` | S* = argmax R_N | {-1,1}^N | Eq. 13 | `skgame.optim.solvers.*` |
| `Parisi` | R_∞ = 0.7632(2-ε) | R | Eq. 13 | `skgame.theory.analytic.R_inf` |
| `SymbolicProof` | identities | SymPy | all closed forms | `skgame.theory.symbolic` |

Auxiliary objects of the code (not in the paper): `Recorder` streams (`RewardRecorder`, `WindowRecorder`,
`SnapshotRecorder`, `TwoTimeRecorder`), `DMFTResult`, `PlannerProblem`, `SolveResult`, `BenchmarkResult`,
`RunContext`, `Pipeline`.
