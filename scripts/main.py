"""
main.py – Build the Jacobi executables with CMake, submit all sweep jobs,
then submit a dependent plot job.  Meant to be run inside a SLURM job
(via submit_pipeline.sh) so nothing computationally heavy runs on the login node.

Usage (from login node):
    sbatch submit_pipeline.sh

Optional environment overrides:
    BUILD_DIR, TIMING_CSV, SLURM_PARTITION, SLURM_ACCOUNT, SLURM_TIME

Dry-run (inspect scripts without submitting):
    python main.py --dry-run
"""

import subprocess
import numpy as np
import os
import sys
import textwrap
from pathlib import Path
from constants_analysis import (
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


def submit_job(script_content: str, dry_run: bool = False, extra_sbatch_args: list[str] | None = None) -> str | None:
    """Write a batch script to a temp file, submit it, return the job-id string."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_job.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    if dry_run:
        print(f"[dry-run]\n{script_content}")
        return None
    cmd = ["sbatch"] + (extra_sbatch_args or []) + [script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    os.remove(script_path)
    return result.stdout.strip().split()[-1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_abs  = os.path.join(script_dir, BUILD_DIR)
    csv_abs    = os.path.abspath(CSV_PATH)

    # ---- 1. Build -------------------------------------------------------
    cmake_build(build_abs, source_dir=script_dir)

    # Absolute paths to executables
    exe = {
        "none":     os.path.join(build_abs, "jacobi"),
        "just_omp": os.path.join(build_abs, "jacobi_just_omp"),
        "omp_mpi":  os.path.join(build_abs, "jacobi_omp_mpi"),
    }

    # ---- 2. Parameter sweeps -------------------------------------------
    sizes_grid           = list(np.linspace(100, 1000, 2).astype(int))
    number_mpi_processes = [1, 2, 4, 8, 16, 32, 64]
    number_omp_threads   = [1, 2, 4, 8, 16, 32, 64]

    job_ids: list[str] = []

    for nx in sizes_grid:
        ny = nx
        script = _make_serial_script(exe["none"], nx, ny, csv_abs)
        jid = submit_job(script, dry_run=dry_run)
        if jid:
            job_ids.append(jid)
            print(f"[none]     nx={nx} ny={ny}  → job {jid}")

        for nthreads in number_omp_threads:
            script = _make_omp_script(exe["just_omp"], nx, ny, nthreads, csv_abs)
            jid = submit_job(script, dry_run=dry_run)
            if jid:
                job_ids.append(jid)
                print(f"[just_omp] nx={nx} ny={ny} threads={nthreads}  → job {jid}")

        for nprocs in number_mpi_processes:
            for nthreads in number_omp_threads:
                script = _make_omp_mpi_script(
                    exe["omp_mpi"], nx, ny, nprocs, nthreads, csv_abs
                )
                jid = submit_job(script, dry_run=dry_run)
                if jid:
                    job_ids.append(jid)
                    print(f"[omp_mpi]  nx={nx} ny={ny} procs={nprocs} threads={nthreads}  → job {jid}")

    # ---- 3. Submit plot job with dependency on ALL sweep jobs -----------
    # SLURM runs it automatically once every sweep job has completed.
    if job_ids and not dry_run:
        dep = "afterok:" + ":".join(job_ids)
        plots_dir = os.path.join(script_dir, "plots")
        plot_script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=jacobi_plot
            #SBATCH --partition={SLURM_PARTITION}
            {_account_line(SLURM_ACCOUNT)}\
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=1
            #SBATCH --time=00:10:00
            #SBATCH --output=jacobi_plot_%j.out

            python {script_dir}/analysis.py
        """)
        jid = submit_job(plot_script, dry_run=dry_run, extra_sbatch_args=[f"--dependency={dep}"])
        print(f"[plot]     → job {jid}  (starts after all {len(job_ids)} sweep jobs)")
    elif dry_run:
        print("[dry-run] Would submit plot job with --dependency=afterok:<all_ids>")

    