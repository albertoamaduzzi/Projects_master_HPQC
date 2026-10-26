"""Step-by-step pictures of how K_N (the SK planner problem) is embedded on Zephyr and Pegasus.

Every figure is computed from the real graphs and the real embeddings (minorminer.busclique,
the algorithm behind DWaveCliqueSampler and the `clique` strategy of skgame.optim.embedding);
nothing is drawn by hand. One function per step, each returns the path it wrote:

  step01_logical_problem     the problem graph K_N: every spin talks to every spin
  step02_hardware            the two hardware graphs, qubits coloured by orientation
  step03_couplers            one qubit and its neighbours: internal / external / odd couplers
  step04_lines               qubits in a row joined by external couplers form a "line"
  step05_one_chain           a chain = one vertical line segment + one horizontal one
  step06_all_chains          the whole clique embedding
  step07_crossbar            why it is all-to-all: verticals cross horizontals
  step08_link_matrix         couplers between every pair of chains (all > 0)
  step09_scaling             chain length and qubits vs N, Zephyr vs Pegasus (+ heuristic)
  step10_defects             largest clique when qubits are missing
  animate                    GIF: chains added one at a time

    python embedding_explanation.py                    # all figures into figures/
    python embedding_explanation.py --steps 5 7 --heuristic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "unlearnable_games_explanation"))
from skgame.optim.embedding import (CliqueEmbedding, HeuristicEmbedding, TOPOLOGIES,  # noqa: E402
                                    chain_stats, is_valid_embedding, _graphs)

dg = _graphs()
FIG = HERE / "figures"
# small graphs for drawings (every qubit visible), production graphs for numbers
SMALL = {"zephyr": dict(shape=(2, 2), N=12), "pegasus": dict(shape=(3,), N=12)}
ORIENT = {0: "#1f77b4", 1: "#d62728"}            # vertical / horizontal qubits
KIND_COLOR = {"internal": "#2ca02c", "external": "#ff7f0e", "odd": "#9467bd"}


# ================================================================ graph geometry ====
def graph(topo: str, shape=None):
    shape = shape or TOPOLOGIES[topo].shape
    return dg.zephyr_graph(*shape) if topo == "zephyr" else dg.pegasus_graph(*shape)


def coordinates(G):
    """linear index -> hardware coordinates. Zephyr (u, w, k, j, z), Pegasus (u, w, k, z):
    u = orientation (0 vertical, 1 horizontal), z = position along the line."""
    fam, args = G.graph["family"], (G.graph["rows"], G.graph["tile"]) if G.graph["family"] == "zephyr" else (G.graph["rows"],)
    conv = (dg.zephyr_coordinates(*args).linear_to_zephyr if fam == "zephyr"
            else dg.pegasus_coordinates(*args).linear_to_pegasus)
    return conv


def line_key(c) -> tuple:
    """Qubits with the same key lie on one line (same orientation and offsets, consecutive z)."""
    return tuple(c[:-1])


def coupler_kind(conv, a, b) -> str:
    ca, cb = conv(a), conv(b)
    if ca[0] != cb[0]:
        return "internal"          # a vertical and a horizontal qubit crossing
    if line_key(ca) == line_key(cb):
        return "external"          # neighbours on the same line
    return "odd"                   # parallel qubits side by side


def layout(G):
    return dg.zephyr_layout(G) if G.graph["family"] == "zephyr" else dg.pegasus_layout(G)


def split_chain(chain, conv):
    """(vertical qubits, horizontal qubits) of a chain."""
    v = [q for q in chain if conv(q)[0] == 0]
    h = [q for q in chain if conv(q)[0] == 1]
    return v, h


def clique(G, N):
    emb = CliqueEmbedding().embed(N, G)
    assert is_valid_embedding(emb, N, G)
    return emb


def _base(ax, G, pos, alpha=0.25, node_size=12):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bbbbbb", width=0.3, alpha=alpha)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color="#cccccc")
    ax.set_axis_off()


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def _suptitle(fig, text):
    fig.suptitle(text, y=1.04, fontsize=12)


def _title(topo, G):
    return f"{topo.capitalize()}  {G.graph['name'].split('(')[0]}{G.graph['name'][G.graph['name'].index('('):]}"


# ========================================================================= steps =====
def step01_logical_problem(N: int = 8):
    G = nx.complete_graph(N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, ax=ax, node_color="#444444", font_color="w", edge_color="#888888", width=0.8)
    ax.set_title(f"Step 1 - the problem: K_{N}\n{N} spins, N(N-1)/2 = {N * (N - 1) // 2} couplings W_ij")
    ax.set_axis_off()
    return _save(fig, "step01_logical_problem.png")


def step02_hardware():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo, SMALL[topo]["shape"])
        conv = coordinates(G)
        pos = layout(G)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=0.3, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=14, node_color=[ORIENT[conv(q)[0]] for q in G])
        deg = max(d for _, d in graph(topo).degree())
        ax.set_title(f"{_title(topo, G)}: {G.number_of_nodes()} qubits\n"
                     f"production {TOPOLOGIES[topo].qpu_family} {TOPOLOGIES[topo].shape}: "
                     f"{graph(topo).number_of_nodes()} qubits, degree {deg}")
        ax.set_axis_off()
    axes[0].plot([], [], "o", color=ORIENT[0], label="vertical qubit (u=0)")
    axes[0].plot([], [], "o", color=ORIENT[1], label="horizontal qubit (u=1)")
    axes[0].legend(loc="lower left", fontsize=8)
    _suptitle(fig, "Step 2 - the hardware: a sparse graph, every qubit has a bounded number of couplers")
    return _save(fig, "step02_hardware.png")


def step03_couplers():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo, (2, 4) if topo == "zephyr" else (4,))
        conv = coordinates(G)
        pos = layout(G)
        dmax = max(d for _, d in G.degree())
        q = next(n for n, d in G.degree() if d == dmax)
        _base(ax, G, pos)
        counts = {}
        for kind, col in KIND_COLOR.items():
            nb = [n for n in G[q] if coupler_kind(conv, q, n) == kind]
            counts[kind] = len(nb)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(q, n) for n in nb], edge_color=col, width=1.6)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nb, node_size=26, node_color=col)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[q], node_size=80, node_color="k")
        ax.set_title(f"{topo.capitalize()}: qubit {conv(q)} has degree {dmax} = "
                     + " + ".join(f"{counts[k]} {k}" for k in KIND_COLOR), fontsize=10)
    for k, c in KIND_COLOR.items():
        axes[1].plot([], [], "-", color=c, lw=2, label=k)
    axes[1].legend(loc="lower left", fontsize=8)
    _suptitle(fig, "Step 3 - three kinds of coupler: internal (vertical x horizontal), "
                 "external (along a line), odd (parallel neighbours)")
    return _save(fig, "step03_couplers.png")


def step04_lines():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo, SMALL[topo]["shape"])
        conv = coordinates(G)
        pos = layout(G)
        lines = {}
        for q in G:
            lines.setdefault(line_key(conv(q)), []).append(q)
        longest = {u: max((k for k in lines if k[0] == u), key=lambda k: len(lines[k])) for u in (0, 1)}
        _base(ax, G, pos)
        for u, k in longest.items():
            qs = sorted(lines[k], key=lambda q: conv(q)[-1])
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(zip(qs, qs[1:])), edge_color=ORIENT[u], width=2.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=qs, node_size=40, node_color=ORIENT[u])
        cross = [(a, b) for a in lines[longest[0]] for b in lines[longest[1]] if G.has_edge(a, b)]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cross, edge_color="k", width=2, style="dashed")
        ax.set_title(f"{topo.capitalize()}: {len(lines)} lines; the two shown meet through "
                     f"{len(cross)} internal coupler(s) (dashed)")
    _suptitle(fig, "Step 4 - a line: qubits of the same orientation and offset, joined end to end by external couplers")
    return _save(fig, "step04_lines.png")


def step05_one_chain():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo, SMALL[topo]["shape"])
        conv = coordinates(G)
        pos = layout(G)
        emb = clique(G, SMALL[topo]["N"])
        i = max(emb, key=lambda k: len(emb[k]))
        v, h = split_chain(emb[i], conv)
        _base(ax, G, pos)
        for qs, u in ((v, 0), (h, 1)):
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e in G.subgraph(qs).edges()],
                                   edge_color=ORIENT[u], width=3)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=qs, node_size=60, node_color=ORIENT[u])
        corner = [(a, b) for a in v for b in h if G.has_edge(a, b)][:1]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=corner, edge_color="k", width=3.5)
        ax.set_title(f"{topo.capitalize()}, K_{SMALL[topo]['N']}: chain of spin {i} = {len(v)} vertical + "
                     f"{len(h)} horizontal qubit(s)\njoined at one internal coupler (black)", fontsize=10)
    _suptitle(fig, "Step 5 - a chain: one vertical line segment + one horizontal line segment,\n"
                 "held together by a strong ferromagnetic coupling (-chain strength)")
    return _save(fig, "step05_one_chain.png")


def step06_all_chains():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo, SMALL[topo]["shape"])
        N = SMALL[topo]["N"]
        emb = clique(G, N)
        draw = dg.draw_zephyr_embedding if topo == "zephyr" else dg.draw_pegasus_embedding
        draw(G, emb, ax=ax, node_size=12, width=0.6, unused_color=(0.85, 0.85, 0.85, 0.6))
        s = chain_stats(emb)
        ax.set_title(f"{topo.capitalize()}: K_{N} on {G.number_of_nodes()} qubits, {s['n_qubits']} used, "
                     f"chain length {s['chain_len_min']}-{s['chain_len_max']}")
    _suptitle(fig, "Step 6 - the clique embedding: one colour per logical spin")
    return _save(fig, "step06_all_chains.png")


def crossbar(G, emb):
    """For each chain i its vertical segment V_i and horizontal segment H_j; returns the matrix
    M[i, j] = number of internal couplers between V_i and H_j (rows/cols in chain order)."""
    conv = coordinates(G)
    parts = {i: split_chain(c, conv) for i, c in emb.items()}
    N = len(emb)
    M = np.zeros((N, N), int)
    for i in range(N):
        for j in range(N):
            M[i, j] = sum(G.has_edge(a, b) for a in parts[i][0] for b in parts[j][1])
    return M


def step07_crossbar(N: int = 24):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo)
        emb = clique(G, N)
        M = crossbar(G, emb)
        conv = coordinates(G)
        # order verticals and horizontals by position on the chip
        ov = sorted(range(N), key=lambda i: line_key(conv(split_chain(emb[i], conv)[0][0])))
        oh = sorted(range(N), key=lambda i: line_key(conv(split_chain(emb[i], conv)[1][0])))
        for x, i in enumerate(ov):
            ax.plot([x, x], [-0.5, N - 0.5], color=ORIENT[0], lw=1, alpha=0.5)
        for y, j in enumerate(oh):
            ax.plot([-0.5, N - 0.5], [y, y], color=ORIENT[1], lw=1, alpha=0.5)
        for x, i in enumerate(ov):
            for y, j in enumerate(oh):
                if M[i, j]:
                    ax.plot(x, y, "o", ms=4 if i != j else 8, color="k" if i == j else "#2ca02c")
        linked_ij = (M + M.T > 0) | np.eye(N, dtype=bool)
        ax.set_title(f"{topo.capitalize()}, K_{N}: {int((M > 0).sum())} of {N * N} (V_i, H_j) are couplers\n"
                     f"pairs {{i,j}} with V_i-H_j or V_j-H_i: {int(linked_ij[np.triu_indices(N, 1)].sum())}"
                     f" / {N * (N - 1) // 2}", fontsize=10)
        ax.set_xlabel("vertical segments V_i (blue), by position")
        ax.set_ylabel("horizontal segments H_j (red), by position")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].plot([], [], "o", color="#2ca02c", label="coupler V_i - H_j: links spins i and j")
    axes[0].plot([], [], "o", color="k", ms=8, label="V_i - H_i: the corner of chain i")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    _suptitle(fig, "Step 7 - why every pair is linked: on one side of the corner diagonal every vertical segment\n"
                 "meets every horizontal one, so for each pair {i, j} V_i-H_j or V_j-H_i is a coupler")
    return _save(fig, "step07_crossbar.png")


def chain_links(G, emb):
    owner = {q: i for i, c in emb.items() for q in c}
    N = len(emb)
    L = np.zeros((N, N), int)
    for a, b in G.edges():
        i, j = owner.get(a), owner.get(b)
        if i is not None and j is not None and i != j:
            L[i, j] += 1
            L[j, i] += 1
    return L


def step08_link_matrix(N: int = 64):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, topo in zip(axes, ("zephyr", "pegasus")):
        G = graph(topo)
        L = chain_links(G, clique(G, N))
        off = L[~np.eye(N, dtype=bool)]
        im = ax.imshow(L, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, label="couplers between chain i and chain j")
        ax.set_title(f"{topo.capitalize()}, K_{N}: min {off.min()} / mean {off.mean():.1f} couplers per pair")
        ax.set_xlabel("chain j")
        ax.set_ylabel("chain i")
    _suptitle(fig, "Step 8 - check: every off-diagonal entry is > 0, so every W_ij has a coupler to live on")
    return _save(fig, "step08_link_matrix.png")


def step09_scaling(heuristic: bool = False, heuristic_Ns=(16, 32, 48, 64), timeout: float = 60.0):
    rows = {}
    for topo in ("zephyr", "pegasus"):
        G = graph(topo)
        big = CliqueEmbedding().max_size(G)
        Ns = list(range(8, big + 1, 8)) + [big]
        st = [chain_stats(clique(G, N)) for N in Ns]
        rows[topo] = (Ns, st, big)
        if heuristic:
            hs = []
            for N in heuristic_Ns:
                emb = HeuristicEmbedding(tries=2, timeout=timeout).embed(N, G)
                if emb:
                    hs.append((N, chain_stats(emb)))
            rows[topo] += (hs,)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    col = {"zephyr": "#1f77b4", "pegasus": "#d62728"}
    for topo, r in rows.items():
        Ns, st, big = r[:3]
        axes[0].plot(Ns, [s["chain_len_mean"] for s in st], "-", color=col[topo], label=f"{topo} clique (max N {big})")
        axes[0].plot(Ns, [s["chain_len_max"] for s in st], ":", color=col[topo])
        axes[1].plot(Ns, [s["n_qubits"] for s in st], "-", color=col[topo], label=f"{topo} clique")
        if heuristic and len(r) > 3:
            hN = [n for n, _ in r[3]]
            axes[0].plot(hN, [s["chain_len_mean"] for _, s in r[3]], "s", color=col[topo], mfc="none",
                         label=f"{topo} minorminer (mean)")
            axes[0].plot(hN, [s["chain_len_max"] for _, s in r[3]], "^", color=col[topo], mfc="none",
                         label=f"{topo} minorminer (max)")
            axes[1].plot(hN, [s["n_qubits"] for _, s in r[3]], "s", color=col[topo], mfc="none")
    axes[0].set_xlabel("N (logical spins)")
    axes[0].set_ylabel("chain length (solid mean, dotted max)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("physical qubits used")
    axes[1].legend(fontsize=7)
    _suptitle(fig, "Step 9 - cost of all-to-all: Zephyr chains ~ N/16 + 2, Pegasus ~ N/12 + 2 (ideal graphs)")
    fig.tight_layout()
    return _save(fig, "step09_scaling.png")


def step10_defects(rates=(0.0, 0.01, 0.02, 0.05, 0.1), seeds=3):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for topo, c in (("zephyr", "#1f77b4"), ("pegasus", "#d62728")):
        G0 = graph(topo)
        means = []
        for r in rates:
            vals = []
            for s in range(seeds if r > 0 else 1):
                rng = np.random.default_rng(s)
                G = G0.copy()
                G.remove_nodes_from(rng.choice(list(G0), int(r * len(G0)), replace=False))
                vals.append(CliqueEmbedding().max_size(G))
            means.append(np.mean(vals))
        ax.plot(np.array(rates) * 100, means, "o-", color=c, label=topo)
    ax.set_xlabel("missing qubits [%] (random)")
    ax.set_ylabel("largest embeddable K_N")
    ax.legend()
    ax.set_title("Step 10 - real chips have dead qubits:\nbusclique searches for intact line segments")
    fig.tight_layout()
    return _save(fig, "step10_defects.png")


def animate(topo: str, fps: float = 1.5):
    from matplotlib.animation import PillowWriter
    G = graph(topo, SMALL[topo]["shape"])
    N = SMALL[topo]["N"]
    emb = clique(G, N)
    conv = coordinates(G)
    pos = layout(G)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=fps)
    path = FIG / f"animation_{topo}.gif"
    FIG.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, path, dpi=90):
        for k in range(N + 1):
            ax.clear()
            _base(ax, G, pos)
            for i in range(k):
                qs = emb[i]
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(G.subgraph(qs).edges()), edge_color=[cmap(i % 20)],
                                       width=2.5)
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=qs, node_size=40, node_color=[cmap(i % 20)])
            if 0 < k:
                v, h = split_chain(emb[k - 1], conv)
                links = [(a, b) for a in emb[k - 1] for j in range(k - 1) for b in emb[j] if G.has_edge(a, b)]
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=links, edge_color="k", width=0.8, style="dotted")
                ax.set_title(f"{topo}: chain {k - 1} = {len(v)} vertical + {len(h)} horizontal qubits;\n"
                             f"dotted: its {len(links)} couplers to the {k - 1} chains already placed")
            else:
                ax.set_title(f"{topo}: hardware graph, K_{N} to embed")
            writer.grab_frame()
        for _ in range(3):
            writer.grab_frame()
    plt.close(fig)
    return path


STEPS = {1: step01_logical_problem, 2: step02_hardware, 3: step03_couplers, 4: step04_lines,
         5: step05_one_chain, 6: step06_all_chains, 7: step07_crossbar, 8: step08_link_matrix,
         9: step09_scaling, 10: step10_defects}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=sorted(STEPS))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--heuristic", action="store_true", help="step 9: also run minorminer (minutes)")
    a = ap.parse_args(argv)
    for s in a.steps:
        p = STEPS[s](heuristic=True) if (s == 9 and a.heuristic) else STEPS[s]()
        print("wrote", p)
    if not a.no_gif:
        for topo in ("zephyr", "pegasus"):
            print("wrote", animate(topo))


if __name__ == "__main__":
    main()
