"""SymPy verification of every closed-form step used in the paper.

Each check returns a dict {name, eq, claim, passed, detail}. Two checks document typos in the
paper (Eq. B36 and Eq. 49) and verify the corrected forms used in this package.
"""
from __future__ import annotations

import sympy as sp

e, a_, b_, x, t, al, beta, et, D0 = sp.symbols("epsilon a b x t alpha beta eta Delta0", real=True)


def _chk(name, eq, claim, expr_zero=None, passed=None, detail=""):
    if passed is None:
        passed = sp.simplify(expr_zero) == 0
    return {"name": name, "eq": eq, "claim": claim, "passed": bool(passed), "detail": str(detail)}


def check_variance():
    a = 1 - e / 2
    b = e / 2
    ups = sp.expand(a**2 + b**2)
    return _chk("variance", "Eq. 9", "N Var J_ij = (1-e/2)^2+(e/2)^2 = 1 - e + e^2/2",
                ups - (1 - e + e**2 / 2), detail=ups)


def check_eta():
    a = 1 - e / 2
    b = e / 2
    # E[J_ij J_ji] = a^2 E[S^2] + b^2 E[A_ij A_ji] = (a^2 - b^2)/N
    num = sp.expand(a**2 - b**2)
    ups = 1 - e + e**2 / 2
    return _chk("eta", "Eq. 10", "corr(J_ij,J_ji) = (1-e)/ups", num / ups - (1 - e) / ups, detail=num)


def check_reward_symmetric_part():
    n = 3
    S = sp.Matrix(sp.symbols("S1:4"))
    Js = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"s{min(i,j)}{max(i,j)}") if i != j else 0)
    Ja = sp.Matrix(n, n, lambda i, j: (sp.Symbol(f"a{min(i,j)}{max(i,j)}") * (1 if i < j else -1)) if i != j else 0)
    J = (1 - e / 2) * Js + (e / 2) * Ja
    lhs = (S.T * J * S)[0]
    rhs = (1 - e / 2) * (S.T * Js * S)[0]
    return _chk("antisymmetric part drops out", "after Eq. 12",
                "sum S J S = (1-e/2) sum S J^S S", sp.expand(lhs - rhs))


def check_Rinf():
    e0 = sp.Symbol("e0", positive=True)
    return _chk("R_inf", "Eq. 13", "(1/N) S^T J S at SK ground state = 2 e0 (1-e/2) = e0 (2-e)",
                2 * e0 * (1 - e / 2) - e0 * (2 - e))


def check_lambda_max():
    ups = 1 - e + e**2 / 2
    eta_ = (1 - e) / ups
    lam = sp.sqrt(ups) * (1 + eta_)
    target = (2 - e) ** 2 / (2 * sp.sqrt(ups))
    return _chk("elliptic-law edge", "Eq. 20",
                "lambda_max = sqrt(ups)(1+eta) = (2-e)^2/(2 sqrt ups)  => T_c = lambda_max",
                sp.simplify(lam - target))


def check_Tc_two_routes():
    """Eq. 49 (+Eq. 50) must reproduce Eq. 20. It does only with a factor beta on the left."""
    ups = 1 - e + e**2 / 2
    b = sp.Symbol("b", positive=True)
    chi = (1 - sp.sqrt(1 - 4 * b**2 * (1 - e))) / (2 * b * (1 - e))
    bc = 2 * sp.sqrt(ups) / (2 - e) ** 2  # 1/T_c from Eq. 20
    corrected = b * (1 - e) * chi - (1 - b * sp.sqrt(ups))
    vals = [0, sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(4, 5)]
    res_c = [sp.N(corrected.subs({e: v}).subs(b, bc.subs(e, v)), 30) for v in vals]
    literal = (1 - e) * chi - (1 - b * sp.sqrt(ups))
    res_l = [sp.N(literal.subs({e: v}).subs(b, bc.subs(e, v)), 15) for v in vals]
    ok = all(abs(r) < 1e-20 for r in res_c) and any(abs(r) > 1e-3 for r in res_l)
    # exact symbolic proof for the corrected form: beta(1-e)chi = (1-sqrt(D))/2, D = 1-4b^2(1-e)
    # => sqrt(D) = 2 b sqrt(ups) - 1  => b = sqrt(ups)/(ups + 1 - e) = 2 sqrt(ups)/(2-e)^2
    proof = sp.simplify(sp.sqrt(ups) / (ups + 1 - e) - bc)
    return _chk("T_c from DMFT stability", "Eq. 49-50 vs Eq. 20",
                "beta(1-e)chi = 1 - beta sqrt(ups) gives T_c(e) of Eq. 20 (paper's Eq. 49 lacks the beta)",
                passed=ok and proof == 0,
                detail=f"residual corrected={[float(r) for r in res_c]}, literal Eq.49={[float(r) for r in res_l]}")


def check_ewma_variance():
    k = sp.Symbol("k", integer=True, nonnegative=True)
    s = sp.summation(al**2 * (1 - al) ** (2 * k), (k, 0, sp.oo))
    s = s.replace(lambda z: isinstance(z, sp.Piecewise), lambda z: z.args[0][0])  # convergent branch
    s = sp.simplify(s)
    return _chk("EWMA noise", "Eq. 18 / A2", "alpha^2 sum_k (1-alpha)^{2k} = alpha/(2-alpha)",
                s - al / (2 - al), detail=s)


def check_continuous_limit():
    """Q(t+alpha) - Q(t) = alpha(-Q + F): Taylor to 2nd order gives alpha/2 Q'' + Q' = -Q + F."""
    Qf = sp.Function("Q")
    F = sp.Function("F")
    tau = sp.Symbol("tau")
    lhs = sp.series(Qf(tau + al), al, 0, 3).removeO() - Qf(tau)
    eq = sp.expand(lhs / al)
    target = Qf(tau).diff(tau) + al / 2 * Qf(tau).diff(tau, 2)
    return _chk("continuous limit", "Eq. 33", "[Q(t+a)-Q(t)]/a = Q' + (a/2) Q'' + O(a^2)",
                sp.simplify(eq.doit() - target.doit()))


def check_sompolinsky_crisanti():
    D = sp.Symbol("D", positive=True)
    y = sp.Symbol("y", positive=True)
    intC = sp.integrate(sp.Rational(2) / sp.pi * sp.asin(y / D0), (y, 0, D0))
    U = -D0**2 / 2 + intC / 2
    sol = sp.solve(sp.simplify(U), D0)
    sol = [s for s in sol if s != 0]
    ok1 = len(sol) == 1 and sp.simplify(sol[0] - (1 - 2 / sp.pi)) == 0
    # linearisation: C ~ (2/pi) Delta/Delta0  =>  Delta'' = (1 - 1/(pi Delta0)) Delta
    rate = 1 - 1 / (sp.pi * (1 - 2 / sp.pi))
    tau1 = 1 / sp.sqrt(rate)
    ok2 = sp.simplify(tau1**2 - (sp.pi - 2) / (sp.pi - 3)) == 0
    return _chk("Sompolinsky-Crisanti", "Eq. 37-39, D5",
                "Delta0 = 1-2/pi from U(Delta0)=0; tau1 = sqrt((pi-2)/(pi-3)) ~ 2.84",
                passed=ok1 and ok2, detail=f"Delta0={sol}, tau1={sp.N(tau1, 6)}")


def check_complexity_forms():
    """Eq. 22 is stationary exactly at Eq. 23; Eq. B36 (-x^2/2) is not (typo)."""
    Phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2
    phi = lambda z: sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi)
    S22 = -et * x**2 / 2 + sp.log(2) + sp.log(Phi(et * x))
    SB36 = -x**2 / 2 + sp.log(2) + sp.log(Phi(et * x))
    cond = x * Phi(et * x) - phi(et * x)  # Eq. 23
    xs = sp.nsolve(cond.subs(et, 1), x, 0.5)
    d22 = sp.N(sp.diff(S22, x).subs({et: 1, x: xs}))
    xs2 = sp.nsolve(cond.subs(et, sp.Rational(1, 2)), x, 0.6)
    dB36 = sp.N(sp.diff(SB36, x).subs({et: sp.Rational(1, 2), x: xs2}))
    d22b = sp.N(sp.diff(S22, x).subs({et: sp.Rational(1, 2), x: xs2}))
    val = sp.N(S22.subs({et: 1, x: xs}))
    ok = abs(d22) < 1e-12 and abs(d22b) < 1e-12 and abs(dB36) > 1e-3 and abs(val - 0.19923) < 1e-4
    zero = sp.simplify(S22.subs(et, 0))
    return _chk("fixed-point complexity", "Eq. 22-23 vs B36",
                "Sigma(1)=0.1992 (Tanaka-Edwards), Sigma(0)=0; Eq. 23 is the stationarity of Eq. 22",
                passed=ok and zero == 0,
                detail=f"Sigma(1)={val}, dS22/dx={d22}, dSB36/dx(eta=1/2)={dB36}")


def check_sigma2_alpha1():
    Phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2
    F = -et * (0 + x**2) + sp.log(2) + sp.log(2 * Phi(et * x) ** 2)
    target = 2 * (-et * x**2 / 2 + sp.log(2) + sp.log(Phi(et * x)))
    return _chk("Sigma_2(alpha=1) = 2 Sigma_1", "Eq. B50",
                "F(r=k=v=0, w=x) = 2 x Eq.22", sp.simplify(sp.expand_log(F - target, force=True)))


def check_two_cycle_jacobian():
    a = 1 - al
    J = sp.Matrix([[-a, 1], [1, -a]])
    inv = 1 / (1 - a**2) * sp.Matrix([[a, 1], [1, a]])
    return _chk("2-cycle Jacobian", "Eq. B37", "(J^alpha)^{-1} = [[1-a,1],[1,1-a]]/(1-(1-a)^2)",
                passed=sp.simplify(J * inv - sp.eye(2)) == sp.zeros(2, 2), detail=sp.factor(J.det()))


def check_chi_quadratic():
    chi = sp.Symbol("chi")
    sol = sp.solve(sp.Eq(chi, beta / (1 - beta * (1 - e) * chi)), chi)
    target = (1 - sp.sqrt(1 - 4 * beta**2 * (1 - e))) / (2 * beta * (1 - e))
    ok = any(sp.simplify(s - target) == 0 for s in sol)
    return _chk("paramagnetic chi", "Eq. 50", "chi = beta/(1-beta(1-e)chi) -> smaller root",
                passed=ok, detail=sol)


ALL_CHECKS = [check_variance, check_eta, check_reward_symmetric_part, check_Rinf, check_lambda_max,
              check_Tc_two_routes, check_ewma_variance, check_continuous_limit,
              check_sompolinsky_crisanti, check_complexity_forms, check_sigma2_alpha1,
              check_two_cycle_jacobian, check_chi_quadratic]


def run_all():
    out = []
    for f in ALL_CHECKS:
        try:
            out.append(f())
        except Exception as exc:  # pragma: no cover
            out.append({"name": f.__name__, "eq": "", "claim": "", "passed": False, "detail": repr(exc)})
    return out
