import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe in HPC environments
import matplotlib.pyplot as plt
import pandas as pd
import os
from constants_analysis import CSV_PATH


def load_results(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the timing CSV; raise a clear error if it does not exist."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Timing CSV not found: {csv_path}\n"
            "Run main.py to generate it."
        )
    df = pd.read_csv(csv_path)
    df["total_workers"] = df["number_omp_threads"] * df["number_mpi_processes"]
    return df


def plot_methods_comparison(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Single figure comparing all three methods (none, just_omp, omp_mpi).

    x-axis : grid size (nx = ny)
    y-axis : total_time (s), averaged over repeated runs
    lines  :
      - none        (serial baseline, 1 worker)
      - just_omp    one line per OMP thread count
      - omp_mpi     one line per (MPI processes × OMP threads) combination
    """
    os.makedirs(output_dir, exist_ok=True)
    col = "total_time"
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    color_idx = 0

    # ---- none (serial baseline) ----
    ndf = df[df["case"] == "none"].groupby("nx")[col].mean().reset_index().sort_values("nx")
    if not ndf.empty:
        ax.plot(ndf["nx"], ndf[col], marker="s", linewidth=2,
                color=colors[color_idx], label="none (serial)")
        color_idx += 1

    # ---- just_omp ----
    odf = df[df["case"] == "just_omp"]
    for nthreads, grp in odf.groupby("number_omp_threads"):
        agg = grp.groupby("nx")[col].mean().reset_index().sort_values("nx")
        ax.plot(agg["nx"], agg[col], marker="o", linewidth=1.5,
                color=colors[color_idx % len(colors)],
                label=f"just_omp  t={nthreads}")
        color_idx += 1

    # ---- omp_mpi ----
    mdf = df[df["case"] == "omp_mpi"]
    for (nprocs, nthreads), grp in mdf.groupby(["number_mpi_processes", "number_omp_threads"]):
        agg = grp.groupby("nx")[col].mean().reset_index().sort_values("nx")
        ax.plot(agg["nx"], agg[col], marker="^", linewidth=1.5, linestyle="--",
                color=colors[color_idx % len(colors)],
                label=f"omp_mpi  p={nprocs} t={nthreads}")
        color_idx += 1

    # ---- gpu ----
    gdf = df[df["case"] == "gpu"]
    if not gdf.empty:
        agg = gdf.groupby("nx")[col].mean().reset_index().sort_values("nx")
        ax.plot(agg["nx"], agg[col], marker="D", linewidth=2, linestyle="-.",
                color=colors[color_idx % len(colors)], label="gpu (OpenMP offload)")
        color_idx += 1

    ax.set_xlabel("Grid size  (nx = ny)")
    ax.set_ylabel("Total execution time (s)")
    ax.set_title("Execution time vs Grid size – all methods")
    ax.legend(title="Method / configuration", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "methods_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_speedup_vs_grid(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Speedup of each parallel configuration relative to the serial baseline
    (none, 1 proc, 1 thread) for the same grid size.

    x-axis : grid size (nx = ny)
    y-axis : speedup  T_serial / T_parallel
    lines  : one per (case, mpi procs, omp threads) combination (excluding none)
    """
    os.makedirs(output_dir, exist_ok=True)
    col = "total_time"

    # Serial reference: none, averaged per nx
    serial = (
        df[df["case"] == "none"]
        .groupby("nx")[col].mean()
        .rename("t_serial")
    )

    if serial.empty:
        print("No 'none' case data found; skipping speedup plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    color_idx = 0

    for case in ["just_omp", "omp_mpi"]:
        cdf = df[df["case"] == case]
        group_cols = (
            ["number_mpi_processes", "number_omp_threads"]
            if case == "omp_mpi"
            else ["number_omp_threads"]
        )
        for keys, grp in cdf.groupby(group_cols):
            agg = grp.groupby("nx")[col].mean().reset_index().sort_values("nx")
            agg = agg.join(serial, on="nx")
            agg["speedup"] = agg["t_serial"] / agg[col]
            if case == "omp_mpi":
                nprocs, nthreads = keys
                label = f"omp_mpi  p={nprocs} t={nthreads}"
                marker = "^"
                ls = "--"
            else:
                nthreads = keys if not isinstance(keys, tuple) else keys[0]
                label = f"just_omp  t={nthreads}"
                marker = "o"
                ls = "-"
            ax.plot(agg["nx"], agg["speedup"], marker=marker, linestyle=ls,
                    linewidth=1.5, color=colors[color_idx % len(colors)], label=label)
            color_idx += 1

    # ---- gpu ----
    gdf = df[df["case"] == "gpu"]
    if not gdf.empty:
        agg = gdf.groupby("nx")[col].mean().reset_index().sort_values("nx")
        agg = agg.join(serial, on="nx")
        agg["speedup"] = agg["t_serial"] / agg[col]
        ax.plot(agg["nx"], agg["speedup"], marker="D", linestyle="-.",
                linewidth=2, color=colors[color_idx % len(colors)],
                label="gpu (OpenMP offload)")
        color_idx += 1

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="serial baseline")
    ax.set_xlabel("Grid size  (nx = ny)")
    ax.set_ylabel("Speedup  $T_{serial} / T_{parallel}$")
    ax.set_title("Speedup over serial baseline vs Grid size")
    ax.legend(title="Method / configuration", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "speedup_vs_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    df = load_results()
    plots_dir = "plots"
    plot_methods_comparison(df, output_dir=plots_dir)
    plot_speedup_vs_grid(df, output_dir=plots_dir)
