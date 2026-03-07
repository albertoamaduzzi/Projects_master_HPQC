import os

CASES_PARALLELISM = ["none", "just_omp", "omp_mpi"]

TIMING_COLUMNS = [
    "fill_boundary_grid_time",
    "fill_grid_without_boundary_time",
    "update_time",
    "check_convergence_time",
    "total_time",
]

# Path to the shared CSV that all executables append to.
# Can be overridden via the TIMING_CSV environment variable.
CSV_PATH = os.environ.get("TIMING_CSV", "timing_results.csv")

# ---- SLURM / run configuration ----
BUILD_DIR = os.environ.get("BUILD_DIR", "build")   # CMake build output directory

SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "regular")
SLURM_ACCOUNT   = os.environ.get("SLURM_ACCOUNT",   "tra26_mhpcqc")
SLURM_TIME      = os.environ.get("SLURM_TIME",      "00:30:00")
SLURM_NODES     = 1            # nodes per job (increase for large MPI counts)