"""Measure the QPU's control error sigma and effective inverse temperature beta_eff from its own reads.

    python -m dwave_utils noise-fit --out results/sk                 # every QPU call with N <= 12
    python -m dwave_utils noise-fit --out results/sk --max-N 16 --include-broken   # slower
    python -m dwave_utils feasibility --topology zephyr --N 16 64 104 --from-fit results/sk

Model of one QPU call (one programming of one instance):
  * the chip holds the logical couplings W_ij / D + delta_ij, where D is the divisor that brought the
    problem into the coupler range (saved with the call) and delta_ij ~ N(0, sigma^2 |E_ij|): each of the
    |E_ij| physical couplers between chains i and j carries an independent error of size sigma
    (programmed units, i.e. fractions of the full range); the errors are FIXED for the whole call;
  * every read is drawn from the Boltzmann distribution of that perturbed problem,
        P(S | delta) = exp(-beta_eff E_delta(S)) / Z_delta ,  E_delta(S) = H(S)/D - sum_{i<j} delta_ij S_i S_j.
Since delta is unknown, the likelihood of the call's reads is the average over delta,
        L(beta, sigma) = E_delta [ prod_reads P(S_r | delta) ],
computed with a Laplace approximation around the most likely delta (Newton's method; accurate with the
hundreds of reads of a call). For N <= max_N all 2^(N-1) states are enumerated (h = 0: S and -S are
equivalent), so Z is exact. Cost grows like 2^N N^4: N <= 12 takes seconds, N = 16 minutes per call.
The log-likelihoods of all calls of one machine are summed on a (beta, sigma) grid; the maximum is the
estimate, and the region within 1.92 of it (profile likelihood) is the 95 % interval.

Identifiability: beta sets how sharply the reads concentrate on low energies; sigma sets how often the
calls prefer a state that is NOT the ground state of the intended instance (each call its own way).
Calls at different chain strengths (calibration scan) have different D and help separate the two.
Limits: flux (field) errors and chain breaks are not modelled (by default only reads with no broken
chain are used); a freeze-out that is not Boltzmann shows up as a lower beta.
Outputs: <out>/calibration/noise_fit.json and noise_fit.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BETAS = np.geomspace(0.3, 100.0, 36)
SIGMAS = np.array([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1])


def all_states(N: int) -> np.ndarray:
    """The 2^(N-1) configurations with S_0 = +1, as a (M, N) float array."""
    M = 1 << (N - 1)
    bits = ((np.arange(M)[:, None] >> np.arange(N - 1)) & 1).astype(np.float64)
    return np.hstack([np.ones((M, 1)), 1.0 - 2.0 * bits])


def _pair_energy(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """-sum_{i<j} A_ij S_i S_j for every row of S (A symmetric, zero diagonal)."""
    return -0.5 * np.einsum("mi,mi->m", S @ A, S)


def _bonds(states: np.ndarray) -> tuple:
    N = states.shape[1]
    iu = np.triu_indices(N, 1)
    return states[:, iu[0]] * states[:, iu[1]], iu                       # (M, pairs): S_i S_j


def call_loglik(W: np.ndarray, D: float, Ecount: np.ndarray, S_obs: np.ndarray, counts: np.ndarray,
                betas=BETAS, sigmas=SIGMAS, **_ignored) -> np.ndarray:
    """log L(beta, sigma) of one call's reads, shape (len(betas), len(sigmas)), integrating the unknown
    coupler errors delta with a Laplace approximation (accurate when the call has many reads):

      g(delta) = beta o.(w + delta) - R log Z(beta (w + delta))      o = sum_r B(S_r), w = (W/D)_pairs
      phi(delta) = g(delta) - sum delta^2 / (2 sigma^2 e)             e = |E_ij|
      log L ~ phi(delta*) - 1/2 log det(I + sigma^2 diag(e) beta^2 R Cov_delta*(B))
    with B(S) = (S_i S_j)_{i<j} and delta* found by Newton's method (warm-started along sigma)."""
    N = len(W)
    states = all_states(N)
    B, iu = _bonds(states)
    S_obs = np.asarray(S_obs, float) * np.sign(np.asarray(S_obs, float)[:, :1])
    idx = ((1.0 - S_obs[:, 1:]) / 2.0).astype(np.int64) @ (1 << np.arange(N - 1))
    counts = np.asarray(counts, float)
    R = counts.sum()
    o = counts @ B[idx]
    w = (W / D)[iu]
    e = np.maximum(Ecount[iu], 1.0)
    out = np.empty((len(betas), len(sigmas)))

    def logZ_mu_cov(theta, need_cov):
        a = B @ theta
        m = a.max()
        p = np.exp(a - m)
        Z = p.sum()
        p /= Z
        mu = p @ B
        cov = (B * p[:, None]).T @ B - np.outer(mu, mu) if need_cov else None
        return m + np.log(Z), mu, cov

    for jb, b in enumerate(betas):
        delta = np.zeros_like(w)
        for js, sg in enumerate(sigmas):
            if sg == 0:
                lz, _, _ = logZ_mu_cov(b * w, False)
                out[jb, js] = b * o @ w - R * lz
                continue
            prec = 1.0 / (sg * sg * e)
            for _ in range(30):                                          # Newton on phi (concave)
                lz, mu, cov = logZ_mu_cov(b * (w + delta), True)
                grad = b * (o - R * mu) - prec * delta
                H = b * b * R * cov + np.diag(prec)                      # -Hessian, positive definite
                step = np.linalg.solve(H, grad)
                phi0 = b * o @ (w + delta) - R * lz - 0.5 * prec @ delta ** 2
                t = 1.0
                while t > 1e-4:                                          # backtracking line search
                    nd = delta + t * step
                    lz1, _, _ = logZ_mu_cov(b * (w + nd), False)
                    if b * o @ (w + nd) - R * lz1 - 0.5 * prec @ nd ** 2 >= phi0 - 1e-12:
                        break
                    t /= 2
                delta = delta + t * step
                if np.max(np.abs(t * step)) < 1e-9:
                    break
            lz, mu, cov = logZ_mu_cov(b * (w + delta), True)
            phi = b * o @ (w + delta) - R * lz - 0.5 * prec @ delta ** 2
            M = np.eye(len(w)) + (sg * sg * e)[:, None] * (b * b * R * cov)
            out[jb, js] = phi - 0.5 * np.linalg.slogdet(M)[1]
    return out


def _calls(out: Path, max_N: int):
    """Every saved QPU call (main run and calibration scan) with N <= max_N: (label, record path)."""
    pats = ["anneal/qpu_*/*/N*_i*.json", "coupling_scan/qpu_*/*/N*_p*_i*.json"]
    for pat in pats:
        for f in sorted(out.glob(pat)):
            if f.name.endswith("_physical.json"):
                continue
            N = int(f.name[1:5])
            if N <= max_N:
                yield f.parent.parent.name, f


def _call_data(out: Path, cfg, f: Path, include_broken: bool):
    import main_simulation as ms
    from skgame.optim.problem import PlannerProblem
    r = json.loads(f.read_text())
    info = r["info"]
    N, key, b = r["N"], r["key"], r["instance"]
    classes = {k: (e, j) for k, e, j in ms._classes(cfg)}
    if key not in classes or "reads" not in r or not info.get("programmed"):
        return None
    eps, J0 = classes[key]
    W = PlannerProblem(ms.ensure_instances(out, cfg, key, eps, J0, N)[b]).W
    D = float(info["programmed"]["auto_scale_divisor"])
    Ecount = np.ones((N, N))
    pf = f.with_name(f.stem + "_physical.npz")
    if pf.exists() and info.get("embedding_used"):
        ph = np.load(pf)
        own = {q: int(i) for i, ch in info["embedding_used"].items() for q in ch}
        Ecount = np.zeros((N, N))
        for x, y in zip(ph["u"], ph["v"]):
            i, j = own[int(x)], own[int(y)]
            if i != j:
                Ecount[i, j] += 1
                Ecount[j, i] += 1
        Ecount = np.maximum(Ecount, 1.0)
    rd = r["reads"]
    keep = [k for k, nb in enumerate(rd["n_broken"]) if include_broken or nb == 0]
    if not keep:
        return None
    S = np.array([[1.0 if ch == "+" else -1.0 for ch in rd["S"][k]] for k in keep])
    cnt = np.array([rd["count"][k] for k in keep], float)
    return W, D, Ecount, S, cnt


def _grid_sum(data, betas, sigmas):
    acc = {}
    for lab, (W, D, Ecount, S, cnt) in data:
        ll = call_loglik(W, D, Ecount, S, cnt, betas, sigmas)
        acc[lab] = acc.get(lab, 0.0) + ll
    return acc


def _refine(betas, sigmas, jb, js, n=21):
    """A finer grid around the coarse maximum (two coarse steps on each side)."""
    b = np.geomspace(betas[max(jb - 2, 0)], betas[min(jb + 2, len(betas) - 1)], n)
    s = np.linspace(sigmas[max(js - 2, 0)], sigmas[min(js + 2, len(sigmas) - 1)], n)
    return b, s


def fit(out: Path, cfg, max_N: int = 12, include_broken: bool = False, betas=BETAS, sigmas=SIGMAS,
        log=print) -> dict:
    """Coarse grid over every machine's calls, then a finer grid around each machine's maximum."""
    out = Path(out)
    data, n_reads = [], {}
    for lab, f in _calls(out, max_N):
        d = _call_data(out, cfg, f, include_broken)
        if d is None:
            continue
        data.append((lab, d))
        n_reads[lab] = n_reads.get(lab, 0) + int(d[4].sum())
    res = {"model": "Boltzmann at beta_eff on the problem with fixed per-call coupler errors N(0, sigma^2 |E_ij|), "
                    "programmed units", "max_N": max_N, "include_broken": include_broken, "machines": {}}
    if not data:
        log(f"[noise-fit] no QPU call with N <= {max_N} in {out}")
        return res
    coarse = _grid_sum(data, betas, sigmas)
    surfaces = {}
    for lab, ll in coarse.items():
        jb, js = np.unravel_index(np.argmax(ll), ll.shape)
        edge = bool(jb in (0, len(betas) - 1) or js == len(sigmas) - 1)
        fb, fs = _refine(betas, sigmas, jb, js)
        fine = _grid_sum([x for x in data if x[0] == lab], fb, fs)[lab]
        kb, ks = np.unravel_index(np.argmax(fine), fine.shape)
        ok_c = coarse[lab] >= fine.max() - 1.92                      # 95 % region on the coarse grid (wide)
        ok_f = fine >= fine.max() - 1.92
        bi = [float(fb[i]) for i in range(len(fb)) if ok_f[i].any()] + \
             [float(betas[i]) for i in range(len(betas)) if ok_c[i].any()]
        si = [float(fs[j]) for j in range(len(fs)) if ok_f[:, j].any()] + \
             [float(sigmas[j]) for j in range(len(sigmas)) if ok_c[:, j].any()]
        topo = "zephyr" if "zephyr" in lab else "pegasus" if "pegasus" in lab else lab
        res["machines"][lab] = {"topology": topo, "beta_eff": float(fb[kb]), "sigma": float(fs[ks]),
                                "beta_95": [min(bi), max(bi)], "sigma_95": [min(si), max(si)],
                                "calls": sum(1 for x in data if x[0] == lab), "reads": n_reads[lab],
                                "at_grid_edge": edge}
        surfaces[lab] = coarse[lab]
        m = res["machines"][lab]
        log(f"[noise-fit] {lab}: beta_eff = {m['beta_eff']:.2f} (95%: {m['beta_95'][0]:.2f}-{m['beta_95'][1]:.2f}), "
            f"sigma = {m['sigma']:.4f} (95%: {m['sigma_95'][0]:.4f}-{m['sigma_95'][1]:.4f}) "
            f"from {m['calls']} calls, {m['reads']} reads" + ("  [at the grid edge]" if edge else ""))
    d = out / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    (d / "noise_fit.json").write_text(json.dumps(res, indent=1))
    _plot(d / "noise_fit.png", surfaces, betas, sigmas)
    log(f"[noise-fit] -> {d / 'noise_fit.json'}, {d / 'noise_fit.png'}  (use: feasibility --from-fit {out})")
    return res


def load_fit(out: Path, topology: str) -> dict | None:
    p = Path(out) / "calibration" / "noise_fit.json"
    if not p.exists():
        return None
    for m in json.loads(p.read_text())["machines"].values():
        if m["topology"] == topology:
            return m
    return None


def _plot(path: Path, acc: dict, betas, sigmas) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(acc), figsize=(5.5 * len(acc), 4.2), squeeze=False)
    for ax, (lab, ll) in zip(axes[0], acc.items()):
        rel = np.maximum(ll - ll.max(), -30)
        im = ax.pcolormesh(sigmas, betas, rel, shading="nearest", cmap="viridis")
        ax.contour(sigmas, betas, rel, levels=[-1.92], colors="w")
        jb, js = np.unravel_index(np.argmax(ll), ll.shape)
        ax.plot(sigmas[js], betas[jb], "r*", ms=12)
        ax.set_yscale("log")
        ax.set_xlabel("sigma (control error, fraction of the coupler range)")
        ax.set_ylabel("beta_eff (programmed units)")
        ax.set_title(f"{lab}: log-likelihood (white: 95 % region)", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
