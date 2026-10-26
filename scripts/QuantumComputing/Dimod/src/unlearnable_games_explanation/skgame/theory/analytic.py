"""Closed-form / quasi-closed-form results of the paper.

Every function cites the equation it implements. Symbolic cross-checks live in symbolic.py.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special
from scipy.stats import norm

from ..model import upsilon, eta

PARISI_E0 = 0.763166726  # -ground-state energy per spin of SK, H = -sum_{i<j} J_ij S_i S_j


# ---------------------------------------------------------------------------- Eq. 13 --
def R_inf(eps):
    return PARISI_E0 * (2.0 - np.asarray(eps, float))


# ---------------------------------------------------------------------------- Eq. 20 --
def Tc(eps):
    """Critical noise where m=0 becomes linearly unstable.

    Elliptic law: eigenvalues of J fill an ellipse with real semi-axis sqrt(ups)(1+eta),
    so lambda_max = sqrt(ups)(1+eta) = (2-eps)^2 / (2 sqrt(ups)); beta_c lambda_max = 1.
    """
    eps = np.asarray(eps, float)
    return 0.5 * (2.0 - eps) ** 2 / np.sqrt(upsilon(eps))


def lambda_max_elliptic(eps):
    u = upsilon(eps)
    return np.sqrt(u) * (1.0 + eta(eps))


# ------------------------------------------------------------------------ Eq. 22, 23 --
def complexity_fp_T0(et: float) -> tuple[float, float]:
    """Annealed fixed-point complexity at beta=inf (Hwang et al. 2019).

    Sigma(eta) = -eta x^2/2 + log 2 + log Phi(eta x),   x Phi(eta x) = Phi'(eta x).
    NOTE: Eq. (B36) of the paper writes -x^2/2; that form is NOT stationary at Eq. (23);
    the consistent form is Eq. (22) (checked: it gives Sigma(1)=0.19923, Tanaka-Edwards).
    Returns (Sigma, x).
    """
    if et <= 0:
        # for eta <= 0 the maximiser is x = 0 at eta = 0 and Sigma = 0; for eta<0 Sigma<=0
        if abs(et) < 1e-14:
            return 0.0, 0.0
    f = lambda x: x * norm.cdf(et * x) - norm.pdf(et * x)
    x = optimize.brentq(f, 0.0, 60.0)
    S = -0.5 * et * x * x + np.log(2.0) + norm.logcdf(et * x)
    return float(S), float(x)


def complexity_fp_T0_eps(eps):
    return np.array([complexity_fp_T0(eta(e))[0] for e in np.atleast_1d(eps)])


# ----------------------------------------------------- Sompolinsky-Crisanti, Eq. 36-39 --
DELTA0 = 1.0 - 2.0 / np.pi
TAU1 = np.sqrt((np.pi - 2.0) / (np.pi - 3.0))


def sc_C_of_Delta(D, D0=DELTA0):
    return (2.0 / np.pi) * np.arcsin(np.clip(D / D0, -1.0, 1.0))


def sompolinsky_crisanti(tau_max: float = 12.0, n: int = 1201):
    """Solve Delta'' = Delta - C(Delta)/2, Delta(0)=1-2/pi, Delta'(0)=0, and return C(tau).

    The trajectory runs along the separatrix of V(D) = -D^2/2 + (1/pi)[...]; numerically it
    is integrated until Delta/Delta0 ~ 1e-3, then continued with the exact asymptote
    Delta ~ A exp(-tau/tau1) matched in value.
    """
    tau = np.linspace(0, tau_max, n)

    def rhs(t, y):
        return [y[1], y[0] - 0.5 * sc_C_of_Delta(y[0])]

    def stop(t, y):
        return y[0] - 2e-3 * DELTA0
    stop.terminal = True

    # the exact separatrix needs Delta'(0)=0; tiny errors blow up exponentially, so we
    # integrate the first-order energy equation instead:  (D')^2/2 = V(D0) - V(D) ... with
    # potential U(D) = -D^2/2 + (1/2) int_0^D C  and E = U(D0) = 0 (this fixes D0 = 1-2/pi).
    def U(D):
        x = D / DELTA0
        intC = (2.0 / np.pi) * DELTA0 * (x * np.arcsin(x) + np.sqrt(1 - x * x) - 1.0)
        return -0.5 * D * D + 0.5 * intC

    # tau(D) = int_D^{D0} dD' / sqrt(-2 U(D'))   (U<=0 on (0,D0))
    D_grid = DELTA0 * np.linspace(1.0, 1e-4, 4000)
    integrand = lambda D: 1.0 / np.sqrt(max(-2.0 * U(D), 1e-300))
    taus = np.zeros_like(D_grid)
    for k in range(1, len(D_grid)):
        val, _ = integrate.quad(integrand, D_grid[k], D_grid[k - 1], limit=200)
        taus[k] = taus[k - 1] + val
    Delta = np.interp(tau, taus, D_grid, right=np.nan)
    # exponential continuation
    last = np.isfinite(Delta)
    t_end, D_end = taus[-1], D_grid[-1]
    Delta[~last] = D_end * np.exp(-(tau[~last] - t_end) / TAU1)
    return tau, sc_C_of_Delta(Delta), Delta


# --------------------------------------------- RS / static DMFT fixed point, Eq. 40-44 --
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(121)
_GH_W = _GH_W / _GH_W.sum()


_Z = np.linspace(-8.5, 8.5, 2401)
_WZ = norm.pdf(_Z) * (_Z[1] - _Z[0])
_WZ /= _WZ.sum()


def _outer_branch_m(beta, k_over_beta, h, n_newton=30):
    """Solve m = tanh(beta (c m + h)), c = k/beta, on the outer (stable) branch sign(m)=sign(h).

    On that branch 1 - k(1-m^2) > 0, so the denominators of Eq. (43) are positive; for k<1 the
    root is unique.
    """
    m = np.where(h >= 0, 1.0, -1.0) * (1.0 - 1e-12)
    for _ in range(n_newton):
        th = np.tanh(beta * (k_over_beta * m + h))
        F = m - th
        dF = 1.0 - beta * k_over_beta * (1.0 - th * th)
        dF = np.maximum(dF, 1e-12)
        m = np.clip(m - F / dF, -1.0, 1.0)
    return m


def _rs_map_H(beta, eps, q, k):
    """(q, k) -> (q', k', chi'), k := beta (1-eps) chi, Eq. 42-44 on the outer branch."""
    u = upsilon(eps)
    h = np.sqrt(max(q, 0.0) * u) * _Z
    m = _outer_branch_m(beta, k / beta, h)
    s = 1.0 - m * m
    den = np.maximum(1.0 - k * s, 1e-12)
    qn = np.sum(_WZ * m * m)
    chi = np.sum(_WZ * beta * s / den)
    return qn, beta * (1.0 - eps) * chi, chi


def static_dmft_fixed_point(beta: float, eps: float, q0: float = 0.5, k0: float = 0.3,
                            tol: float = 1e-11, maxit: int = 4000):
    """Solve Eq. 42-44: q = <m^2>, chi = <beta(1-m^2)/(1-beta(1-eps)chi(1-m^2))>,
    m = tanh(beta(1-eps)chi m + beta sqrt(q ups) z). Returns (q, chi).

    Branch choice: for each z the outer (stable) solution with sign(m)=sign(z) (only matters
    when k=beta(1-eps)chi>1, i.e. at low T). Damped iteration in (q, k).
    Coincides with the RS solution of the NMFE (Bray, Sompolinsky, Yu 1986) at eps=0.
    """
    q, k = q0, k0
    for _ in range(60):
        qn, kn, chi = _rs_map_H(beta, eps, q, k)
        q, k = q + 0.5 * (qn - q), k + 0.5 * (kn - k)
    def F(x):
        qn, kn, _ = _rs_map_H(beta, eps, x[0], x[1])
        return [qn - x[0], kn - x[1]]
    sol = optimize.root(F, [q, k], method="hybr", tol=tol)
    if sol.success and sol.x[0] >= -1e-9:
        q, k = sol.x
    else:
        for _ in range(maxit):
            qn, kn, _ = _rs_map_H(beta, eps, q, k)
            q, k = q + 0.5 * (qn - q), k + 0.5 * (kn - k)
    q = max(q, 0.0)
    chi = _rs_map_H(beta, eps, q, k)[2]
    return float(q), float(chi)


def static_dmft_curve(betas, eps: float):
    """Continuation of (q, chi) of Eq. 42-44 along increasing beta (decreasing T)."""
    order = np.argsort(betas)
    out = np.zeros((len(betas), 2))
    q = 0.5
    for j in order[::-1]:          # from low T (large beta) upward: q>0 branch is attractive
        q, chi = static_dmft_fixed_point(float(betas[j]), eps, q0=max(q, 1e-2), k0=0.3)
        out[j] = q, chi
    return out


def chi_near_Tc(beta: float, eps: float) -> float:
    """Paramagnetic (q->0) susceptibility, Eq. (50): beta(1-eps)chi^2 - chi + beta = 0."""
    if abs(1 - eps) < 1e-14:
        return beta
    disc = max(1.0 - 4.0 * beta**2 * (1.0 - eps), 0.0) if eps < 1 else 1.0 - 4.0 * beta**2 * (1.0 - eps)
    return (1.0 - np.sqrt(disc + 0j)).real / (2.0 * beta * (1.0 - eps))


def inverse_spectral_density_omega0(beta: float, eps: float) -> float:
    """1/<|Q1(w=0)|^2> = |1 - beta(1-eps) G_hat(0)|^2 - beta^2 ups   (Eq. 48 at w=0).

    With G_hat(0) = chi of Eq. (50) (paramagnetic branch). Negative => static fixed point
    unstable (Fig. 17b). Setting it to zero gives the criterion beta(1-eps)chi = 1 - beta sqrt(ups);
    NOTE the paper's Eq. (49) omits the factor beta on the left; only with it does one recover
    T_c(eps) of Eq. (20) (see symbolic.check_Tc_two_routes).
    """
    chi = chi_near_Tc(beta, eps)
    return (1.0 - beta * (1.0 - eps) * chi) ** 2 - beta**2 * upsilon(eps)


def Tc_from_stability(eps: float) -> float:
    """Solve beta(1-eps) chi(beta) = 1 - beta sqrt(ups) (corrected Eq. 49 + Eq. 50); return 1/beta_c."""
    u = upsilon(eps)
    f = lambda b: b * (1.0 - eps) * chi_near_Tc(b, eps) - (1.0 - b * np.sqrt(u))
    b_lo = 1e-6
    if (1 - eps) > 1e-14:
        b_hi = 0.5 / np.sqrt(1 - eps)          # edge of the real paramagnetic branch
        if abs(f(b_hi)) < 1e-10:               # eps = 0: root sits exactly on the edge
            return 1.0 / b_hi
    else:
        b_hi = 1e4
    b = optimize.brentq(f, b_lo, b_hi, xtol=1e-14)
    return 1.0 / b


# ------------------------------------------------------------------ Appendix A, Eq. 18 --
def ewma_noise_bound(alpha):
    """Upper bound on the variance of the EWMA estimate of m: alpha/(2-alpha)."""
    alpha = np.asarray(alpha, float)
    return alpha / (2.0 - alpha)


def ewma_noise_fixed_point(alpha, m):
    """Exact stationary variance at a fixed point: alpha/(2-alpha) (1-m^2)  ~ alpha/2 (1-m^2) (A4)."""
    return ewma_noise_bound(alpha) * (1.0 - np.asarray(m) ** 2)


# ------------------------------------------------- Sec. V: L=2 vs exponentially long --
def eps_c_estimate():
    """eps_c ~ 0.8 (Hwang et al.): where e^{aN} beats e^{N Sigma_2}. Quoted, not derived."""
    return 0.8
