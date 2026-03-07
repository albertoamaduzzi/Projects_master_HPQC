import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe in HPC environments
import matplotlib.pyplot as plt
import pandas as pd
import os
from constants_analysis import CASES_PARALLELISM, TIMING_COLUMNS, CSV_PATH


def load_results(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the timing CSV; raise a clear error if it does not exist."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Timing CSV not found: {csv_path}\n"
            "Run main.py to generate it."
        )
    df = pd.read_csv(csv_path)
    # Derived convenience column: total workers
    df["total_workers"] = df["number_omp_threads"] * df["number_mpi_processes"]
    # Human-readable label: case + #mpi
    df["label"] = df.apply(
        lambda r: f"{r['case']} (mpi={r['number_mpi_processes']})", axis=1
    )
    return df


def plot_time_statistic(
    df: pd.DataFrame,
    output_dir: str = ".",
    grid_filter: tuple | None = None,
):
    """
    For every timing column produce one figure.
    x-axis  : number of OMP threads
    lines   : one per (case, number_mpi_processes) combination
    Figures are saved as PNG files in *output_dir*.

    Parameters
    ----------
    df           : DataFrame returned by load_results()
    output_dir   : directory where PNGs will be saved
    grid_filter  : optional (nx, ny) tuple to restrict the data;
                   if None all grid sizes are averaged.
    """
    os.makedirs(output_dir, exist_ok=True)

    if grid_filter is not None:
        nx, ny = grid_filter
        df = df[(df["nx"] == nx) & (df["ny"] == ny)].copy()
        if df.empty:
            print(f"No data for grid ({nx},{ny}). Skipping plots.")
            return

    for col in TIMING_COLUMNS:
        if col not in df.columns:
            print(f"Column '{col}' not in data, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # Group by label, then average over repeated runs of same configuration
        for label, grp in df.groupby("label"):
            agg = (
                grp.groupby("number_omp_threads")[col]
                .mean()
                .reset_index()
                .sort_values("number_omp_threads")
            )
            ax.plot(
                agg["number_omp_threads"],
                agg[col],
                marker="o",
                label=label,
            )

        ax.set_xlabel("OMP threads per process")
        ax.set_ylabel("Time (s)")
        title_suffix = (
            f"  [grid {grid_filter[0]}x{grid_filter[1]}]" if grid_filter else ""
        )
        ax.set_title(f"{col}{title_suffix}")
        ax.legend(title="case (mpi procs)")
        ax.grid(True, linestyle="--", alpha=0.5)

        png_name = f"{col}{'_' + str(grid_filter[0]) + 'x' + str(grid_filter[1]) if grid_filter else ''}.png"
        out_path = os.path.join(output_dir, png_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_all_grids(df: pd.DataFrame, output_dir: str = "plots"):
    """Produce one set of plots per unique (nx, ny) combination in the data."""
    for (nx, ny), _ in df.groupby(["nx", "ny"]):
        plot_time_statistic(df, output_dir=output_dir, grid_filter=(nx, ny))


if __name__ == "__main__":
    df = load_results()
    plot_all_grids(df)

        