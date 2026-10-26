"""
main.py – Build the Jacobi executables with CMake, run the full parameter
sweep directly inside the current SLURM allocation, then generate plots.

The previous approach of submitting individual sub-jobs cannot work on
Leonardo because the lrd_all_serial QOS only allows MaxJobsPU=2 (two
queued jobs per user at a time).  Running everything sequentially inside
one pipeline job sidesteps the limit entirely.

Usage (from login node):
    sbatch submit_pipeline.sh

Dry-run (print what would be executed without running):
    python main.py --dry-run
"""

import subprocess
import time
import numpy as np
import os
import sys
from pathlib import Path
from Projects_master_HPQC.scripts.ParallelComputing.Jacobi.constants_analysis import BUILD_DIR, CSV_PATH


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def cmake_build(build_dir: str, source_dir: str = ".") -> None:
    # Wipe any stale build directory so changed compiler flags take effect.
    build_path = Path(build_dir)
    if build_path.exists():
        import shutil
        shutil.rmtree(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)
    print(f"[cmake] Configuring in {build_dir} ...")
    subprocess.run(
        ["cmake", os.path.abspath(source_dir),
         "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_GPU_OFFLOAD=ON"],
        cwd=build_dir, check=True,
    )
    print("[cmake] Building ...")
    subprocess.run(["cmake", "--build", ".", "--parallel"], cwd=build_dir, check=True)
    print("[cmake] Build complete.")


# ---------------------------------------------------------------------------
# Direct execution helpers
# ---------------------------------------------------------------------------

def run_config(label: str, cmd: list[str], env: dict, dry_run: bool = False) -> None:
    """Run one Jacobi configuration, printing elapsed time."""
    if dry_run:
        omp = env.get("OMP_NUM_THREADS", "?")
        print(f"[dry-run] {label}  OMP={omp}  cmd={' '.join(cmd)}")
        return
    print(f"[run] {label} ...", flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, env=env, check=True)
    print(f"[run] {label} done in {time.monotonic() - t0:.1f}s", flush=True)

# Helper to check for GPU availability. We want to skip GPU runs if no device is present
def gpu_is_available() -> bool:
    """Return True only when at least one NVIDIA GPU is reachable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True,
        )
        return r.returncode == 0 and "GPU 0" in r.stdout
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_abs  = os.path.join(script_dir, BUILD_DIR)
    csv_abs    = os.path.abspath(CSV_PATH)

    # ---- 1. Build -------------------------------------------------------
    cmake_build(build_abs, source_dir=script_dir)

    exe = {
        "none":     os.path.join(build_abs, "jacobi"),
        "just_omp": os.path.join(build_abs, "jacobi_just_omp"),
        "omp_mpi":  os.path.join(build_abs, "jacobi_omp_mpi"),
        "gpu":      os.path.join(build_abs, "jacobi_gpu"),
    }

    # ---- 2. Clear previous results for a clean run ----------------------
    if not dry_run and os.path.exists(csv_abs):
        os.remove(csv_abs)
        print(f"[info] Cleared previous CSV: {csv_abs}")

    base_env = {**os.environ, "TIMING_CSV": csv_abs}

    # ---- 3. Parameter grid (all configs stay within 8 CPUs) ------------
    sizes_grid           = [100, 150, 200, 250, 300, 350, 400]
    number_omp_threads   = [1, 2, 4, 8]
    number_mpi_processes = [1, 2, 4, 8]

    # --- 3a. Serial (no parallelism) ------------------------------------
    for nx in sizes_grid:
        env = {**base_env, "OMP_NUM_THREADS": "1"}
        run_config(
            f"none     nx={nx}",
            [exe["none"], str(nx), str(nx)],
            env, dry_run,
        )

    # --- 3b. OpenMP only -------------------------------------------------
    for nx in sizes_grid:
        for nthreads in number_omp_threads:
            env = {**base_env, "OMP_NUM_THREADS": str(nthreads)}
            run_config(
                f"just_omp nx={nx} t={nthreads}",
                [exe["just_omp"], str(nx), str(nx)],
                env, dry_run,
            )

    # --- 3c. Hybrid MPI + OpenMP ----------------------------------------
    # mpirun --oversubscribe lets us launch more ranks than queued task-slots.
    for nx in sizes_grid:
        for nprocs in number_mpi_processes:
            for nthreads in number_omp_threads:
                if nprocs * nthreads > 8:   # keep within 8-CPU node budget
                    continue
                env = {**base_env, "OMP_NUM_THREADS": str(nthreads)}
                cmd = [
                    "mpirun", "-n", str(nprocs), "--oversubscribe",
                    exe["omp_mpi"], str(nx), str(nx),
                ]
                run_config(
                    f"omp_mpi  nx={nx} p={nprocs} t={nthreads}",
                    cmd, env, dry_run,
                )

    # --- 3d. GPU offload (OpenMP target) --------------------------------
    # Requires a compiler with GPU offload support (NVHPC, Clang, GCC+plugin).
    # On Leonardo, submit to the GPU partition (boost_usr_prod) with
    #   #SBATCH --partition=boost_usr_prod  --gres=gpu:1
    gpu_exe = exe["gpu"]


    if dry_run:
        for nx in sizes_grid:
            env = {**base_env, "OMP_NUM_THREADS": "1"}
            run_config(f"gpu      nx={nx}", [gpu_exe, str(nx), str(nx)], env, dry_run)
    elif not os.path.isfile(gpu_exe):
        print(f"[warn] GPU executable not found: {gpu_exe}  (skipping GPU runs)")
    elif not gpu_is_available():
        print("[warn] No GPU detected on this node (nvidia-smi found no device). "
              "Skipping GPU runs – submit via submit_gpu_pipeline.sh to boost_usr_prod.")
    else:
        for nx in sizes_grid:
            env = {**base_env, "OMP_NUM_THREADS": "1"}
            run_config(
                f"gpu      nx={nx}",
                [gpu_exe, str(nx), str(nx)],
                env, dry_run,
            )

    # ---- 4. Generate plots ----------------------------------------------
    if not dry_run:
        from Projects_master_HPQC.scripts.ParallelComputing.Jacobi.analysis import load_results, plot_methods_comparison, plot_speedup_vs_grid
        plots_dir = os.path.join(script_dir, "plots")
        df = load_results(csv_abs)
        plot_methods_comparison(df, output_dir=plots_dir)
        plot_speedup_vs_grid(df, output_dir=plots_dir)
        print(f"[plots] All figures saved to {plots_dir}/")
        print("[info]  Re-run with --dry-run to preview commands.")
