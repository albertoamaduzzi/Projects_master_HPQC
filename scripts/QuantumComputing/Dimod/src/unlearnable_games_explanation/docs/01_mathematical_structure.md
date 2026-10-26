# 1. Mathematical structure of the SK-game

Reference: J. Garnier-Brun, M. Benzaquen, J.-P. Bouchaud, *Unlearnable Games and "Satisficing" Decisions: A Simple Model for a Complex World*, arXiv:2312.12252 (v2, Jan 2024). Equation numbers below are the paper's.

Notation: indices $i,j,k\in\{1,\dots,N\}$ label agents, $t,s\in\mathbb N$ label time, overline $\overline{\cdot}$ = average over the disorder $J$, $\langle\cdot\rangle$ = average over the learning noise / initial conditions. No summation convention: every sum is written.

---

## 1.1 Data of the model

| symbol | space | meaning |
|---|---|---|
| $N$ | $\mathbb N$ | number of agents |
| $\alpha\in(0,1]$ | | memory-loss rate (EWMA weight), memory time $\tau_\alpha\sim 1/\alpha$ |
| $\beta\in[0,\infty]$ | | intensity of choice (inverse "temperature" $T=1/\beta$) |
| $\varepsilon\in[0,2]$ | | non-reciprocity: $0$ symmetric, $1$ independent, $2$ antisymmetric |
| $J_0,\ J_d$ | $\mathbb R$ | mean (cooperative) coupling, self-coupling ("habit") |
| $J\in\mathbb R^{N\times N}$ | random | interaction matrix (quenched) |

State at time $t$: incentives $Q(t)\in\mathbb R^N$. Derived: intentions $m(t)\in[-1,1]^N$, decisions $S(t)\in\{-1,1\}^N$, rewards $R(t)\in\mathbb R^N$.

## 1.2 Interaction ensemble (Eqs. 8–11)

$$
J_{ij}=\Big(1-\frac{\varepsilon}{2}\Big)J^S_{ij}+\frac{\varepsilon}{2}J^A_{ij}+\frac{J_0}{N}+J_d\,\delta_{ij},\qquad
J^S_{ij}=J^S_{ji},\ J^A_{ij}=-J^A_{ji},\ \ \overline{(J^S_{ij})^2}=\overline{(J^A_{ij})^2}=\frac{\sigma^2}{N},
$$
independent for $i<j$, zero diagonal for $J^S,J^A$. Set $\sigma=1$; write $a=1-\varepsilon/2$, $b=\varepsilon/2$.

**Variance** (Eq. 9). For $i\ne j$, $J^S_{ij}$ and $J^A_{ij}$ are independent and centred:
$$
N\,\overline{J_{ij}^2}=N\big(a^2\overline{(J^S_{ij})^2}+2ab\,\overline{J^S_{ij}J^A_{ij}}+b^2\overline{(J^A_{ij})^2}\big)
=a^2+b^2
=1-\varepsilon+\frac{\varepsilon^2}{4}+\frac{\varepsilon^2}{4}
=1-\varepsilon+\frac{\varepsilon^2}{2}\equiv\upsilon(\varepsilon).
$$

**Reciprocity** (Eq. 10). Using $J^S_{ji}=J^S_{ij}$, $J^A_{ji}=-J^A_{ij}$:
$$
N\,\overline{J_{ij}J_{ji}}=N\,\overline{(aJ^S_{ij}+bJ^A_{ij})(aJ^S_{ij}-bJ^A_{ij})}=a^2-b^2=\Big(1-\frac{\varepsilon}{2}\Big)^2-\frac{\varepsilon^2}{4}=1-\varepsilon,
\qquad
\eta=\frac{\overline{J_{ij}J_{ji}}}{\overline{J_{ij}^2}}=\frac{1-\varepsilon}{\upsilon(\varepsilon)}.
$$
$\eta=1,0,-1$ at $\varepsilon=0,1,2$. (Code: `model.upsilon`, `model.eta`; SymPy: `check_variance`, `check_eta`.)

## 1.3 The learning map (Eqs. 1–4)

For every $i$ and $t$:
$$
\begin{aligned}
m_i(t)&=\tanh\big(\beta Q_i(t)\big), &&\text{(2)}\\
\mathbb P[S_i(t)=\pm1]&=\tfrac12\big(1\pm m_i(t)\big)\quad\text{independently in } i, &&\text{(1)}\\
R_i(t)&=\sum_{j=1}^N J_{ij}S_j(t), &&\text{(4)}\\
Q_i(t+1)&=(1-\alpha)Q_i(t)+\alpha R_i(t). &&\text{(3)}
\end{aligned}
$$
This is a Markov chain on $\mathbb R^N$ (random dynamical system driven by i.i.d. uniforms). At $\beta=\infty$, $S_i=m_i=\operatorname{sign}Q_i$ and the map is deterministic and piecewise affine.

**Unrolling the recursion.** Iterating (3) $t$ times:
$$
Q_i(t)=(1-\alpha)^tQ_i(0)+\alpha\sum_{s=0}^{t-1}(1-\alpha)^{t-1-s}R_i(s)
=(1-\alpha)^tQ_i(0)+\sum_{j}J_{ij}\,\underbrace{\alpha\sum_{s=0}^{t-1}(1-\alpha)^{t-1-s}S_j(s)}_{\tilde m^\alpha_j(t-1)} .
$$
Hence, dropping the transient $(1-\alpha)^tQ_i(0)$, $m_i(t+1)=\tanh\big(\beta\sum_jJ_{ij}\tilde m^\alpha_j(t)\big)$ (Eq. 5) with the EWMA estimate $\tilde m^\alpha_j(t)=\alpha\sum_{s\le t}(1-\alpha)^{t-s}S_j(s)$ (Eq. 6). Offline learning (Eq. 7) replaces $S_j(s)$ by $m_j(s)$. (Test: `test_ewma_of_rewards`.)

**Online noise (Eq. 18, App. A).** With $\xi_j(s)=S_j(s)-m_j(s)$, $\langle\xi_j(s)\rangle=0$, $\langle\xi_j(s)\xi_j(s')\rangle=(1-m_j(s)^2)\delta_{ss'}$:
$$
\Big\langle\Big(\alpha\sum_{s\le t}(1-\alpha)^{t-s}\xi_j(s)\Big)^2\Big\rangle
=\alpha^2\sum_{k\ge0}(1-\alpha)^{2k}\big(1-m_j^2\big)
\le\alpha^2\sum_{k\ge0}(1-\alpha)^{2k}=\frac{\alpha^2}{1-(1-\alpha)^2}=\frac{\alpha^2}{2\alpha-\alpha^2}=\frac{\alpha}{2-\alpha}.
$$
So online ≈ offline up to fluctuations $O(\sqrt\alpha)$, and since $1-m^2\sim\beta^{-2}$ deep in the ordered phase, $O(\sqrt\alpha/\beta)$ (SymPy: `check_ewma_variance`).

**Continuous-time limit (Eq. 33).** Put $\tau=\alpha t$ and $Q(\tau)\equiv Q_t$. Then $Q_{t+1}=Q(\tau+\alpha)$ and Taylor-expanding to second order:
$$
\frac{Q(\tau+\alpha)-Q(\tau)}{\alpha}=\dot Q(\tau)+\frac{\alpha}{2}\ddot Q(\tau)+O(\alpha^2)
\quad\Longrightarrow\quad
\frac{\alpha}{2}\ddot Q+\dot Q=-Q+\sum_j J_{ij}S_j .
$$
(SymPy: `check_continuous_limit`.) The $\frac\alpha2\ddot Q$ term is formally sub-leading but controls the $\alpha^{-1/2}$ cycle period (Sec. VII C).

**Special cases.** $\alpha=1,\beta=\infty$: $S_i(t+1)=\operatorname{sign}\sum_jJ_{ij}S_j(t)$, the parallel Hopfield/SK zero-temperature dynamics (Eq. 24). $J=J_0/N$ only: Curie–Weiss game, $m=\tanh(\beta J_0 m)$.

## 1.4 Welfare and the planner's problem (Eqs. 12–13)

$$
\mathcal R_N(S)=\frac1N\sum_i S_iR_i=\frac1N\sum_{i,j}S_iJ_{ij}S_j .
$$
Antisymmetric part drops: $\sum_{ij}S_iJ^A_{ij}S_j=\sum_{ij}S_jJ^A_{ji}S_i=-\sum_{ij}S_iJ^A_{ij}S_j\Rightarrow 0$. So
$$
\mathcal R_N(S)=\Big(1-\frac\varepsilon2\Big)\frac1N\sum_{i,j}S_iJ^S_{ij}S_j=-\Big(1-\frac\varepsilon2\Big)\frac{2}{N}H_{SK}(S),\qquad H_{SK}=-\sum_{i<j}J^S_{ij}S_iS_j .
$$
The planner maximises $\mathcal R_N$ = finds the SK ground state. With $e_0=\lim -\min H_{SK}/N=0.763166\ldots$ (Parisi),
$$
\mathcal R_\infty=2e_0\Big(1-\frac\varepsilon2\Big)=0.7632\,(2-\varepsilon),\qquad \mathcal R_N\simeq\mathcal R_\infty-\frac{A}{N^{2/3}} .
$$
This is the object of the optimiser benchmark (`skgame.optim`).

## 1.5 Fixed points, Nash equilibria, NMFE (Eq. 19)

For $\alpha\to0$ the dynamics is $m_i(t+1)=\tanh(\beta\sum_jJ_{ij}m_j(t))$, fixed points
$$
m^\star_i=\tanh\Big(\beta\sum_jJ_{ij}m^\star_j\Big)\qquad\text{(Naive Mean-Field Equations / quantal-response equilibria).}
$$
At $\beta=\infty$: $S^\star_i=\operatorname{sign}(\sum_jJ_{ij}S^\star_j)$, i.e. $e_i=S^\star_iR^\star_i=|\sum_jJ_{ij}S^\star_j|>0$ (Eq. 16): **every pure Nash equilibrium of the game is a 1-flip-stable state of $H$ when $\varepsilon=0$**.

**Paramagnet and $T_c$ (Eq. 20).** Linearise at $m=0$: $\delta m_i(t+1)=\beta\sum_jJ_{ij}\delta m_j(t)$. Unstable iff $\beta\,\operatorname{Re}\lambda_{\max}(J)>1$. For $J$ with $N\overline{J_{ij}^2}=\upsilon$, correlation $\eta$, the elliptic law gives eigenvalues uniform in the ellipse with real semi-axis $\sqrt\upsilon(1+\eta)$, so
$$
\lambda_{\max}=\sqrt\upsilon\Big(1+\frac{1-\varepsilon}{\upsilon}\Big)=\frac{\upsilon+1-\varepsilon}{\sqrt\upsilon},\qquad
\upsilon+1-\varepsilon=2-2\varepsilon+\frac{\varepsilon^2}{2}=\frac{(2-\varepsilon)^2}{2},
\qquad T_c(\varepsilon)=\frac{(2-\varepsilon)^2}{2\sqrt{\upsilon(\varepsilon)}} .
$$
$T_c(0)=2$, $T_c(1)=1/\sqrt2$, $T_c(2)=0$.

## 1.6 Complexity (Eqs. 21–23, Sec. V, App. B)

$\Sigma=\lim_N\frac1N\log\mathcal N_J$; annealed version $\frac1N\log\overline{\mathcal N_J}$. At $\beta=\infty$ (Hwang et al. 2019):
$$
\Sigma(\eta)=\max_x\Big\{-\frac{\eta x^2}{2}+\log2+\log\Phi(\eta x)\Big\},\qquad
\partial_x:\ -\eta x+\eta\frac{\Phi'(\eta x)}{\Phi(\eta x)}=0\iff x\,\Phi(\eta x)=\Phi'(\eta x)\ \ \text{(Eq. 23)}.
$$
$\Sigma(1)=0.19923$ (Tanaka–Edwards), $\Sigma(0)=\log2+\log\frac12=0$.

> **Typo found (Eq. B36).** The paper's App. B writes $-\frac12x^2$ instead of $-\frac\eta2x^2$. Only the form of Eq. (22) is stationary at Eq. (23) (SymPy `check_complexity_forms`: $\partial_x$ of Eq. 22 at the root of Eq. 23 is $10^{-16}$, of Eq. B36 is $-0.31$ at $\eta=1/2$).

**2-cycles with memory.** Eq. (B40) has six saddle variables. Under the time-translation ansatz of a 2-cycle, $\hat V(1,1)=\hat V(2,2)=v$, $\hat V(1,2)=\hat V(2,1)=w$, $r=i\hat R(1,2)$, $k=i\hat K(1,2)$, and with $a=1-\alpha$, $D=1-a^2$, Eq. (B38) gives
$$
\Gamma_1=\frac{\eta}{D}\big[w+Sv+a(v+Sw)\big]=\Gamma_2\equiv\Gamma_S,
$$
so the bivariate CDF is evaluated on the diagonal, $\Psi_2(\Gamma,\Gamma;\rho)=\Phi(z)-2T\big(z,\sqrt{(1-\rho)/(1+\rho)}\big)$ (Owen's $T$), with $z=\Gamma_S/\sqrt{C_{11}}$, $\rho=C_{12}/C_{11}$ from Eq. (B39):
$$
\Sigma_2(\alpha,\eta)=\operatorname*{saddle}_{r,k,v,w}\Big\{rk-\eta(v^2+w^2)+\log2+\log\sum_{S=\pm1}e^{Sk}\Psi_2(\Gamma_S,\Gamma_S;\rho_S)\Big\}.
$$
At $\alpha=1$: $r=k=v=0$, $w=x_{\rm FP}$ ⇒ $\Sigma_2=2\Sigma$ (Eq. B50). Numerically (continuation in $\alpha$) $\Sigma_2\to\Sigma$ and $1-C(1)=1+r\to0$ as $\alpha\to0$ — the coalescence of Fig. 10, here obtained down to $\alpha=0.02$.

## 1.7 Dynamical mean-field theory (Sec. VI, App. C)

**Cavity step, expanded.** Add agent $0$. Its field on $i$ shifts $h_i\to h_i+J_{i0}m_0$; linear response of the others: $m_i(t)\to m_i(t)+\sum_{s<t}\sum_j\chi_{ij}(t,s)J_{j0}m_0(s)$. Feeding back into agent 0:
$$
\sum_iJ_{0i}m_i(t)=\underbrace{\sum_iJ_{0i}m_i^{\setminus0}(t)}_{\phi(t)\ \text{Gaussian}}+\sum_{s<t}\sum_{i,j}J_{0i}\chi_{ij}(t,s)J_{j0}\,m_0(s).
$$
Diagonal $i=j$ dominates: $\sum_iJ_{0i}J_{i0}\chi_{ii}\to N\,\overline{J_{0i}J_{i0}}\,G=(1-\varepsilon)G$; variance of $\phi$: $\sum_i\overline{J_{0i}^2}\,\langle m_im_i\rangle=\upsilon\,C$. Online noise adds $\upsilon(1-q)\delta_{ts}$. Result (this package's convention):
$$
Q(t+1)=(1-\alpha)Q(t)+\alpha\Big[(1-\varepsilon)\sum_{s<t}G(t,s)m(s)+\phi(t)+h(t)\Big],\quad
\langle\phi(t)\phi(s)\rangle=\upsilon\big[C(t,s)+(1-q(t))\delta_{ts}\big],
$$
$C(t,s)=\langle m(t)m(s)\rangle$, $G(t,s)=\langle\partial m(t)/\partial h(s)\rangle$. (The paper writes $\alpha^2(1-\varepsilon)G$ with $G$ defined w.r.t. a field without the factor $\alpha$; App. C notes $G\to\alpha G$.)

**Response without finite differences (Novikov).** Because $\phi$ and $h$ enter identically and $\phi$ is Gaussian with covariance $K$:
$$
\langle m(t)\phi(s)\rangle=\sum_{u}K(s,u)\Big\langle\frac{\partial m(t)}{\partial\phi(u)}\Big\rangle=\sum_{u<t}K(s,u)G(t,u)
\ \Rightarrow\ G(t,{:}t)=K[{:}t,{:}t]^{-1}\langle m(t)\phi({:}t)\rangle .
$$
The Cholesky factor of a leading block of $K$ is the leading block of ${\rm chol}(K)$, so all causal solves cost $O(T^3)$ (`theory.dmft.DMFTSolver`).

**$\varepsilon=1$, $\alpha\to0$ (Eqs. 36–39).** Memory term vanishes: $\dot Q=-Q+\phi$, $\langle\phi\phi\rangle=\frac12C$. With $\Delta(\tau)=\langle Q(t+\tau)Q(t)\rangle$: $\ddot\Delta=\Delta-\frac12C$, and for Gaussian $Q$ with $m={\rm sign}\,Q$, $C=\frac2\pi\arcsin(\Delta/\Delta_0)$. Write $\ddot\Delta=-U'(\Delta)$ with
$$
U(\Delta)=-\frac{\Delta^2}{2}+\frac12\int_0^\Delta C,\qquad
\int_0^{\Delta_0}\frac2\pi\arcsin\frac{y}{\Delta_0}dy=\frac2\pi\Delta_0\Big(\frac\pi2-1\Big).
$$
A monotone decay from rest at $\Delta_0$ to $0$ requires $U(\Delta_0)=U(0)=0$:
$$
-\frac{\Delta_0^2}{2}+\frac{\Delta_0}{\pi}\Big(\frac\pi2-1\Big)=0\ \Rightarrow\ \Delta_0=1-\frac2\pi .
$$
Linearising at small $\Delta$: $C\simeq\frac{2}{\pi}\frac{\Delta}{\Delta_0}$, $\ddot\Delta=\Big(1-\frac{1}{\pi\Delta_0}\Big)\Delta=\Big(1-\frac{1}{\pi-2}\Big)\Delta=\frac{\pi-3}{\pi-2}\Delta$, so
$$
C(\tau)\propto e^{-\tau/\tau_1},\qquad \tau_1=\sqrt{\frac{\pi-2}{\pi-3}}=2.8395 .
$$
The full curve is obtained exactly by quadrature, $\tau(\Delta)=\int_\Delta^{\Delta_0}d\Delta'/\sqrt{-2U(\Delta')}$ (no shooting instability).

## 1.8 Static DMFT, RS solution and its instability (Eqs. 40–50)

Time-independent ansatz $\phi=\sqrt{q\upsilon}\,z$, $\chi=\int G$:
$$
m(z)=\tanh\big(\beta(1-\varepsilon)\chi\,m(z)+\beta\sqrt{q\upsilon}\,z\big),\quad q=\langle m^2\rangle_z,\quad
\chi=\Big\langle\frac{\beta(1-m^2)}{1-\beta(1-\varepsilon)\chi(1-m^2)}\Big\rangle_z .
$$
For $k\equiv\beta(1-\varepsilon)\chi>1$ the equation for $m(z)$ has three roots near $z=0$; this package takes the outer (stable) branch ${\rm sign}\,m={\rm sign}\,z$ (denominators then positive). The paper does not state its branch choice, so the low-$T$ part of the static curve in Fig. 17a is branch-dependent.

**Paramagnetic susceptibility** ($q\to0$, $m\simeq\beta(\ldots)$): $\chi=\beta/(1-\beta(1-\varepsilon)\chi)\iff\beta(1-\varepsilon)\chi^2-\chi+\beta=0\Rightarrow\chi=\frac{1-\sqrt{1-4\beta^2(1-\varepsilon)}}{2\beta(1-\varepsilon)}$ (Eq. 50).

**Instability.** Perturbation spectral density at $\omega=0$ (Eq. 48): $\langle|\hat Q_1(0)|^2\rangle^{-1}=|1-\beta(1-\varepsilon)\hat G(0)|^2-\beta^2\upsilon$. Setting it to zero with $\hat G(0)=\chi$:
$$
\beta(1-\varepsilon)\chi=1-\beta\sqrt\upsilon
\iff \frac{1-\sqrt{1-4\beta^2(1-\varepsilon)}}{2}=1-\beta\sqrt\upsilon
\iff \sqrt{1-4\beta^2(1-\varepsilon)}=2\beta\sqrt\upsilon-1
$$
$$
\iff 1-4\beta^2(1-\varepsilon)=4\beta^2\upsilon-4\beta\sqrt\upsilon+1
\iff \beta=\frac{\sqrt\upsilon}{\upsilon+1-\varepsilon}=\frac{2\sqrt\upsilon}{(2-\varepsilon)^2}=\frac{1}{T_c(\varepsilon)} .
$$

> **Typo found (Eq. 49).** The paper writes $(1-\varepsilon)\chi=1-\beta\sqrt\upsilon$ (no $\beta$ on the left). With that literal form one does **not** recover Eq. (20) (at $\varepsilon=0$ the residual at $\beta_c=1/2$ is $0.5$); with the factor $\beta$ one recovers it exactly for every $\varepsilon$ (SymPy `check_Tc_two_routes`, test `test_Tc_values_and_two_routes`).

## 1.9 Aging (Eqs. 51–52)

$C(t_w,t_w+t)=C_{\rm relax}(t)+C_{\rm aging}(t,t_w)$, $C_{\rm relax}\sim t^{-x}$, $C_{\rm aging}=\mathcal C(t/t_w)$ (simple aging) for $\varepsilon<\varepsilon^\star\approx\varepsilon_{\rm RM}\approx0.47$; time-translation invariance for $\varepsilon>\varepsilon^\star$.

## 1.10 The whole structure in one diagram

$$
\underbrace{(N,\alpha,\beta,\varepsilon)}_{\rm Params}
\xrightarrow{\ \rm sample\ }J
\xrightarrow{\ \rm simulate\ }\{Q,m,S\}_t
\xrightarrow{\ \rm observables\ }(\mathcal R_N,e,C,q,M,C^\times,\omega_0)
$$
$$
(\upsilon,\eta)\xrightarrow{\ \rm elliptic\ }T_c,\quad
(\upsilon,\eta)\xrightarrow{\ \rm annealed\ }\Sigma,\Sigma_L,\quad
(\upsilon,\eta,\alpha,\beta)\xrightarrow{\ \rm DMFT\ }(C,G,q)\xrightarrow{\varepsilon=1,\alpha\to0}\tau_1,\quad
J\xrightarrow{\rm sym}W\xrightarrow{\rm optimise}S^\star\xrightarrow{}\mathcal R_N^{\rm planner}\approx\mathcal R_\infty
$$
The machine-readable version of this diagram is `skgame/mathcat.py` → `category/category.json` (key `math`), consumed by the Catlab module.
