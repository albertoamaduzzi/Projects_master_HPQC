"""Plot of the real `qpu --check` output of 2026-10-27 (Leap project of the user, 1000 reads,
annealing_time 20 us): estimated QPU access time per call vs N, cumulative budget.
The numbers are copied from the terminal output; no QPU time was used to produce them."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
Ns = np.arange(8, 161, 8)
zephyr_ms = [167, 176, 186, 191, 199, 207, 210, 213, 213, 213, 213, 213, 213]           # Advantage2_system1, N <= 104
pegasus_ms = [157, 197, 214, 229, 236, 248, 259, 264, 273, 281, 283, 285, 288, 289, 291, 292, 292, 292, 292, 292]
CALLS_PER_N = 4 * 5                                                                     # classes x instances
BUDGET = 180.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(Ns[:len(zephyr_ms)], zephyr_ms, "o-", label="Advantage2_system1 (Zephyr), largest N = 104")
axes[0].plot(Ns, pegasus_ms, "s-", label="Advantage_system4 (Pegasus), largest N = 177")
axes[0].set_xlabel("N (logical spins)")
axes[0].set_ylabel("estimated QPU access time per call [ms]")
axes[0].set_title("solver timing model, 1000 reads x 20 us anneal:\nreadout grows with the qubits used, then saturates", fontsize=9)
axes[0].legend(fontsize=8)
z = np.cumsum(np.array(zephyr_ms) * CALLS_PER_N) / 1e3
p = np.cumsum(np.array(pegasus_ms) * CALLS_PER_N) / 1e3
axes[1].plot(Ns[:len(z)], z, "o-", label=f"Zephyr: {z[-1]:.1f} s")
axes[1].plot(Ns, p, "s-", label=f"Pegasus: {p[-1]:.1f} s")
axes[1].plot(Ns, p + np.concatenate([z, np.full(len(Ns) - len(z), z[-1])]), "k-", lw=2,
             label=f"both: {p[-1] + z[-1]:.1f} s")
axes[1].axhline(BUDGET, color="r", ls="--", label="monthly budget 180 s")
axes[1].set_xlabel("campaign done up to N")
axes[1].set_ylabel("cumulative QPU access time [s]")
axes[1].set_title("preset full: 4 classes x 5 instances per N", fontsize=9)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "qpu_check_budget.png", dpi=140)
print(f"zephyr {z[-1]:.1f} s, pegasus {p[-1]:.1f} s, both {z[-1] + p[-1]:.1f} s")
