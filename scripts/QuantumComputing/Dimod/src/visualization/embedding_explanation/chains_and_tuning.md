# Chains: how they work and how to tune the QPU for better SK solutions

Companion to [README.md](README.md) (how the chains are built). This document covers what the chains do to the problem the QPU actually solves, and which knobs improve the solutions. Every knob listed here is a field of `AnnealSettings` (`skgame/optim/backends.py`) or of the campaign config, so it can be scanned with `main_simulation.py`.

## 1. From the logical problem to the physical one

The logical problem is $H(S)=-\sum_{i<j}W_{ij}S_iS_j$ with $S_i=\pm1$, $i=1..N$. With an embedding $\{C_i\}$, the QPU is programmed with an Ising problem on the physical qubits $s_q=\pm1$:

$$
H_{\rm phys}(s)=\underbrace{\sum_{i<j}\ \sum_{(q,q')\in E_{ij}}\frac{-W_{ij}}{|E_{ij}|}\,s_qs_{q'}}_{\text{logical couplings, split over the couplers between } C_i \text{ and } C_j}
\;\underbrace{-\;c\sum_i\sum_{(q,q')\in T_i}s_qs_{q'}}_{\text{chain couplings}} ,
$$

where:
* $E_{ij}$ is the set of hardware couplers between chain $C_i$ and chain $C_j$: on average $|E_{ij}|\approx1.2$ on Zephyr and $1.1$ on Pegasus at $N=160$;
* $T_i$ is the set of couplers inside chain $C_i$: a path, so $|C_i|-1$ of them, i.e. 11 on Zephyr and about 13.8 on Pegasus at $N=160$;
* $c>0$ is the **chain strength**.

If every chain is intact ($s_q=S_i$ for all $q\in C_i$), the chain term is a constant and

$$
H_{\rm phys}=\sum_{i<j}\ |E_{ij}|\cdot\frac{-W_{ij}}{|E_{ij}|}\,S_iS_j+\text{const}=H(S)+\text{const},
$$

so the low-energy states of the physical problem are the low-energy states of the SK problem. A **broken chain** (qubits of $C_i$ disagree) costs $2c$ for each disagreeing chain coupler, but it may lower the logical part. If $c$ is too small, broken states win.

## 2. Chain strength

The default is D-Wave's *uniform torque compensation* (`dwave.embedding.chain_strength.uniform_torque_compensation`):

$$
c=p\;J_{\rm rms}\,\sqrt{\bar d},\qquad J_{\rm rms}=\sqrt{\frac{1}{|E|}\sum_{i<j}W_{ij}^2},\qquad \bar d=\text{average degree of the problem graph}.
$$

**Why that form.** A chain cut in two halves feels, from its neighbours, a random "torque": a sum of about $\bar d/2$ terms of size $J_{\rm rms}$ with random signs. By the central limit theorem its typical size is $\propto J_{\rm rms}\sqrt{\bar d}$, and the chain coupling has to beat it.

**For SK.** $W_{ij}\sim\mathcal N(0,1/N)$, so $J_{\rm rms}\simeq1/\sqrt N$, and $\bar d=N-1$. Therefore

$$
c\simeq p\cdot\frac{1}{\sqrt N}\cdot\sqrt{N-1}\simeq p .
$$

With the default $p=1.414$ the chain strength is about 1.4 *whatever N is*. Measured on real instances: $N=64$: $J_{\rm rms}=0.1253$, $c=1.414\cdot0.1253\cdot\sqrt{63}=1.414\cdot0.1253\cdot7.94=1.406$. $N=160$: $J_{\rm rms}=0.0784$, $c=1.414\cdot0.0784\cdot\sqrt{159}=1.414\cdot0.0784\cdot12.61=1.398$.

## 3. The price of the chains: energy resolution

The QPU accepts couplings in $[-1,1]$, and with `auto_scale` the whole $H_{\rm phys}$ is divided by its largest coefficient. The chain couplings ($c\approx1.4$) are far larger than the logical ones ($\max|W_{ij}|\approx0.29$ at $N=160$), so the scale factor is fixed by the chains:

$$
\lambda=\frac{1}{\max(c,\ \max|W_{ij}|/|E_{ij}|)}=\frac{1}{c}=\frac{1}{1.398}=0.715 .
$$

The logical couplers are therefore programmed at (measured, $N=160$):

| | rms of logical $|J|$ before scaling | after scaling ($\times\lambda=0.715$) |
|---|---|---|
| Zephyr | 0.0702 | $0.0702\cdot0.715=0.0502$ |
| Pegasus | 0.0739 | $0.0739\cdot0.715=0.0528$ |

So the information that defines the SK instance lives in couplings of about **0.05** on a scale of 1. The QPU's analogue control errors (integrated control errors, of order $10^{-2}$ on each $J$) are then a sizeable fraction of each coupling. Moreover, the energy gaps between the ground state and the first metastable states of SK shrink with $N$. This is the central trade-off:

* **$c$ too small**: chains break, and the read-out spins are not a faithful $S$;
* **$c$ too large**: the logical problem is squeezed into a small part of the range and drowned in noise.

Shorter chains help on both sides. They have fewer places to break, and they need less $c$ to stay intact. This is the physical reason to expect **Zephyr (chains of 12 at $N=160$) to beat Pegasus (chains of 15)**.

## 4. Reading a broken chain

When the qubits of $C_i$ disagree, a rule decides $S_i$ (`chain_break_method`):

| method | rule | cost |
|---|---|---|
| `majority_vote` (default) | $S_i=\operatorname{sign}\sum_{q\in C_i}s_q$, random on ties | free |
| `weighted_random` | $S_i=+1$ with probability equal to the fraction of $+1$ qubits | free, adds noise |
| `discard` | drop every read with a broken chain | wastes reads |
| `minimize_energy` | fix the intact chains, then choose the broken ones greedily to minimise $H$ | small CPU cost, usually best |

`chain_break_fraction_mean` (the fraction of broken chains, averaged over reads) is saved for every call. Values above a few percent mean $c$ is too low.

## 5. Knobs that improve the solutions, and how to use them here

All of these act on the same physical problem. Change them through `init` (or by editing `config.json` before the QPU stage), and compare runs in separate campaign directories.

1. **Topology: Zephyr vs Pegasus.** Shorter chains, more couplers per pair. Run both backends on the same instances: `backends = ["qpu:zephyr:clique", "qpu:pegasus:clique"]` (the default).
2. **Embedding strategy.** For $K_N$ the clique embedding is the right one: uniform, shortest chains (README step 9). `minorminer` is there as a baseline (`qpu:zephyr:minorminer`).
3. **Chain-strength prefactor $p$** (`chain_strength_prefactor`). D-Wave recommends scanning $p\in[0.5,2]$. For SK, $c\simeq p$ (§2), so the scan is directly a scan of the energy scale. Watch two quantities: `chain_break_fraction` rises as $p$ falls, `p_success` peaks in between.
4. **Chain-break method** (`chain_break_method`). `minimize_energy` usually recovers more ground states than `majority_vote` at the same $p$, which allows a lower $p$ and therefore better resolution.
5. **Annealing time** (`annealing_time`, µs). Longer anneals are closer to adiabatic and give the chains time to order. The cost per read is $t_a+t_{\rm readout}+t_{\rm delay}$, so going from 20 to 100 µs changes it from $20+100+20=140$ µs to $100+100+20=220$ µs. Compare with TTS$_{99}$, not with the success probability alone.
6. **Number of reads** (`num_reads`). It does not change the physics, only the statistics. `p_success` per read is what TTS uses.
7. **Polishing** (`polish=True`, default). Each distinct read is relaxed by greedy single-flip descent on $W$ (classical, milliseconds). It repairs the last few wrong spins left by chain breaks and analogue noise. The campaign reports both `p_success` (raw QPU) and `p_success_polished`, so the QPU's own contribution stays visible.
8. **Spin-reversal (gauge) transforms.** Flipping a random subset of qubits ($s_q\to-s_q$, with the signs of $h$ and $J$ adjusted) leaves the problem unchanged but averages out the qubit-dependent biases. Use `dwave.preprocessing.SpinReversalTransformComposite` around the child sampler. It costs one programming time per transform.
9. **Anneal offsets.** These shift the start of the anneal per qubit, to synchronise chains of different lengths. Clique chains have (almost) equal length, so this matters less here than for heuristic embeddings.
10. **Extended J range.** Advantage solvers accept chain couplings down to $-2$ (`extended_j_range`). Putting the chains there instead of at $-1$ doubles the room left for the logical couplings ($\lambda\to2/c$), at the price of more cross-talk. `DWaveCliqueSampler` does this automatically. `FixedEmbeddingComposite` (used by the backends, so that the Zephyr and Pegasus runs are identical apart from the hardware) does not.
11. **Parallel embeddings for small N.** For $N\lesssim40$ a whole chip can hold many copies of $K_N$. Running them together (`minorminer.utils.parallel_embeddings`) multiplies the reads per programming cycle. This is useful for statistics at small N, not for quality.

### Suggested protocol
1. Choose $N$ at the edge of what works, e.g. the smallest $N$ where `p_success` of the default run drops below 0.5.
2. Scan $p\in\{0.5,0.75,1,1.41,2\}$ × {`majority_vote`, `minimize_energy`} on 2–3 instances. Each point costs $n_{\rm inst}\cdot t_{\rm call}$, e.g. $3\cdot0.155\,{\rm s}=0.47$ s of QPU time.
3. Fix the best $(p,\text{method})$ and scan `annealing_time` $\in\{5,20,100,500\}$ µs, comparing TTS$_{99}$.
4. Run the full campaign on both topologies with the chosen settings.

## 6. How the comparison is made fair

The `qpu:zephyr:clique` and `qpu:pegasus:clique` backends:
* solve **the same instances** (`instances/<class>/N….npz`);
* use the same `AnnealSettings` (reads, annealing time, prefactor $p$, chain-break method);
* each get their own chain strength from the same formula, computed on their own embedding;
* use embeddings computed on the **real working graph** of the solver used, cached in `embeddings/<solver>/clique/` and recorded in every result file.

`analyze` then writes `summary/backend_comparison.csv` and `summary/backend_comparison.png`, with the winner per (problem class, N): higher `p_success`, ties broken by lower residual. It also prints how many points each topology wins.
