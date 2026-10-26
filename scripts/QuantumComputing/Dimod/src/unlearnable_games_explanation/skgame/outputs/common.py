"""Shared helpers for output classes."""
from __future__ import annotations

import numpy as np

from ..model import GameParams, SKGame, WindowRecorder, INF


def beta_of(T: float) -> float:
    return INF if T == 0 else 1.0 / T


def simulate_window(N, alpha, eps, beta=INF, R=8, t_eq=1000, window=500, stride=1, seed=0, J0=0.0,
                    Jd=0.0, store_m=False, store_S=True, store_e=False, J=None, Q0=None, extra=()):
    """Run t_eq + window steps; return (game, window dict)."""
    p = GameParams(N=N, alpha=alpha, beta=beta, eps=eps, J0=J0, Jd=Jd)
    g = SKGame(p, R=R, seed=seed, J=J, Q0=Q0)
    w = WindowRecorder(t_eq, window, stride=stride, store_m=store_m, store_S=store_S, store_e=store_e)
    g.run(t_eq + window, [w, *extra])
    return g, w.result()


def eps_colors(eps_list, cmap="viridis"):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    n = len(eps_list)
    return [cm(0.05 + 0.85 * k / max(n - 1, 1)) for k in range(n)]


def save_fig(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = folder / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(str(p))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def ci95(x, axis=0):
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    return 1.96 * np.nanstd(x, axis=axis) / np.sqrt(np.maximum(n, 1))
