"""
main.py – Build the Jacobi HPC executables with CMake and submit one SLURM
job per (case, grid_size, omp_threads, mpi_processes) combination.

Environment variables (all optional):
  BUILD_DIR        – CMake build directory (default: build)
  TIMING_CSV       – Path to the shared CSV for results (default: timing_results.csv)
  SLURM_PARTITION  – Partition to use (default: regular)
  SLURM_ACCOUNT    – Account / project code  (default: empty)
  SLURM_TIME       – Wall-time per job        (default: 00:30:00)
"""

from analysis import plot_all_grids, load_results
import subprocess
import numpy as np
import os
import sys
import textwrap
import time
from pathlib import Path
from constants_analysis import (
    CASES_PARALLELISM,
    BUILD_DIR,
    CSV_PATH,
    SLURM_PARTITION,
    SLURM_ACCOUNT,
    SLURM_TIME,
    SLURM_NODES,
)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def cmake_build(build_dir: str, source_dir: str = ".") -> None:
    """Configure and build all targets with CMake."""
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    print(f"[cmake] Configuring in {build_dir} ...")
    subprocess.run(
        ["cmake", os.path.abspath(source_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir, check=True,
    )
    print("[cmake] Building ...")
    subprocess.run(["cmake", "--build", ".", "--parallel"], cwd=build_dir, check=True)
    print("[cmake] Build complete.")


# ---------------------------------------------------------------------------
# SLURM job script generation and submission
# ---------------------------------------------------------------------------

def _account_line(account: str) -> str:
    return f"#SBATCH --account={account}\n" if account else ""


def _make_serial_script(exe: str, nx: int, ny: int, csv_abs: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=jacobi_none_{nx}x{ny}
        #SBATCH --partition={SLURM_PARTITION}
        {_account_line(SLURM_ACCOUNT)}\
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=1
        #SBATCH --time={SLURM_TIME}
        #SBATCH --output=slurm_jacobi_none_{nx}x{ny}_%j.out

        export TIMING_CSV="{csv_abs}"
        export OMP_NUM_THREADS=1
        {exe} {nx} {ny}
    """)


def _make_omp_script(exe: str, nx: int, ny: int, omp_threads: int, csv_abs: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=jacobi_omp_{nx}x{ny}_t{omp_threads}
        #SBATCH --partition={SLURM_PARTITION}
        {_account_line(SLURM_ACCOUNT)}\
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={omp_threads}
        #SBATCH --time={SLURM_TIME}
        #SBATCH --output=slurm_jacobi_omp_{nx}x{ny}_t{omp_threads}_%j.out

        export TIMING_CSV="{csv_abs}"
        export OMP_NUM_THREADS={omp_threads}
        {exe} {nx} {ny}
    """)


def _make_omp_mpi_script(
    exe: str, nx: int, ny: int, mpi_procs: int, omp_threads: int, csv_abs: str
) -> str:
    total_cores = mpi_procs * omp_threads
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=jacobi_ompmpi_{nx}x{ny}_p{mpi_procs}_t{omp_threads}
        #SBATCH --partition={SLURM_PARTITION}
        {_account_line(SLURM_ACCOUNT)}\
        #SBATCH --nodes={SLURM_NODES}
        #SBATCH --ntasks={mpi_procs}
        #SBATCH --cpus-per-task={omp_threads}
        #SBATCH --time={SLURM_TIME}
        #SBATCH --output=slurm_jacobi_ompmpi_{nx}x{ny}_p{mpi_procs}_t{omp_threads}_%j.out

        export TIMING_CSV="{csv_abs}"
        export OMP_NUM_THREADS={omp_threads}
        srun --ntasks={mpi_procs} --cpus-per-task={omp_threads} {exe} {nx} {ny}
    """)


def submit_job(script_content: str, dry_run: bool = False) -> str | None:
    """Write a temporary batch script and submit it; return the job-id string."""
    script_path = "tmp_slurm_job.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    if dry_run:
        print(f"[dry-run] Would submit:\n{script_content}")
        return None
    result = subprocess.run(
        ["sbatch", script_path], capture_output=True, text=True, check=True
    )
    os.remove(script_path)
    job_id = result.stdout.strip().split()[-1]
    return job_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    # ---- 1. Build -------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_abs  = os.path.join(script_dir, BUILD_DIR)
    cmake_build(build_abs, source_dir=script_dir)

    # Absolute paths to executables
    exe = {
        "none":     os.path.join(build_abs, "jacobi"),
        "just_omp": os.path.join(build_abs, "jacobi_just_omp"),
        "omp_mpi":  os.path.join(build_abs, "jacobi_omp_mpi"),
    }

    csv_abs = os.path.abspath(CSV_PATH)

    # ---- 2. Parameter sweeps -------------------------------------------
    sizes_grid          = list(np.linspace(100, 1000, 2).astype(int))
    number_mpi_processes = [1, 2, 4, 8, 16, 32, 64]
    number_omp_threads  = [1, 2, 4, 8, 16, 32, 64]

    job_ids: list[str] = []

    for nx in sizes_grid:
        ny = nx   # square grids
        # --- case: none (serial) ---
        script = _make_serial_script(exe["none"], nx, ny, csv_abs)
        jid = submit_job(script, dry_run=dry_run)
        if jid:
            job_ids.append(jid)
            print(f"[none]     nx={nx} ny={ny}  → job {jid}")

        # --- case: just_omp ---
        for nthreads in number_omp_threads:
            script = _make_omp_script(exe["just_omp"], nx, ny, nthreads, csv_abs)
            jid = submit_job(script, dry_run=dry_run)
            if jid:
                job_ids.append(jid)
                print(f"[just_omp] nx={nx} ny={ny} threads={nthreads}  → job {jid}")

        # --- case: omp_mpi ---
        for nprocs in number_mpi_processes:
            for nthreads in number_omp_threads:
                script = _make_omp_mpi_script(
                    exe["omp_mpi"], nx, ny, nprocs, nthreads, csv_abs
                )
                jid = submit_job(script, dry_run=dry_run)
                if jid:
                    job_ids.append(jid)
                    print(
                        f"[omp_mpi]  nx={nx} ny={ny} "
                        f"procs={nprocs} threads={nthreads}  → job {jid}"
                    )

    # ---- 3. Wait for all jobs, then plot --------------------------------
    if job_ids and not dry_run:
        print(f"\nSubmitted {len(job_ids)} jobs. Waiting for completion ...")
        # Poll sacct every 30 s until all jobs have finished
        pending = set(job_ids)
        while pending:
            time.sleep(30)
            finished = set()
            for jid in pending:
                r = subprocess.run(
                    ["sacct", "-j", jid, "--format=State", "--noheader", "-P"],
                    capture_output=True, text=True,
                )
                states = {s.strip() for s in r.stdout.splitlines() if s.strip()}
                terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}
                if states and states.issubset(terminal):
                    finished.add(jid)
            pending -= finished
            if pending:
                print(f"  {len(pending)} jobs still running …")
        print("All jobs finished.")

    # ---- 4. Generate plots ----------------------------------------------
    if os.path.isfile(csv_abs):
        print("Generating plots ...")
        df = load_results(csv_abs)
        plots_dir = os.path.join(script_dir, "plots")
        plot_all_grids(df, output_dir=plots_dir)
        print(f"Plots saved to {plots_dir}")
    else:
        print(f"CSV not found at {csv_abs}; skipping plots.")

    