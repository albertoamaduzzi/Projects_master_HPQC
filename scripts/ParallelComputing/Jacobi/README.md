# Jacobi — 2D Iterative Solver (HPC exam project)

2D Laplace equation solved by Jacobi iteration on an `Nx × Ny` grid, with fixed
Dirichlet boundaries (`TOP=LEFT=RIGHT=1.0`, `BOTTOM=10.0`), convergence
tolerance `1e-4`, cap `MAX_ITER=500000` (see `constants_jacobi.h`). Built for
the Leonardo cluster (CINECA), SLURM account `tra26_mhpcqc`. Full writeup:
`../../latex/exam_hpc_hpqc.pdf` (source: `../../latex/description_analysis.tex`),
titled *"Jacobi Iteration on a 2D Grid: Serial, OpenMP, and Hybrid MPI+OpenMP
Parallelisation"* (March 2026).

## Implementations

Four executables share the CLI `./jacobi_* NX NY`, all logic in a companion
`.h` so each `.c` contains only `main`:

- **`jacobi.c` / `jacobi.h`** — serial baseline. Timed with `clock_gettime(CLOCK_MONOTONIC)`.
- **`jacobi_just_omp.c` / `jacobi_just_omp.h`** — OpenMP loop parallelism over
  `update` and the residual reduction. Convergence checked every `CHECK_EVERY`
  (=10) iterations to cut reduction overhead. Timed with `omp_get_wtime()`.
- **`jacobi_omp_mpi.c` / `jacobi_omp_mpi.h`** — hybrid MPI+OpenMP. 1D row-strip
  domain decomposition across ranks (`MPI_THREAD_FUNNELED`). Non-blocking halo
  exchange (`MPI_Isend`/`Irecv`) overlapped with interior-row updates;
  boundary rows updated after `MPI_Waitall`. Per-rank times reduced to rank 0
  with `MPI_MAX` (bottleneck view).
- **`jacobi_gpu.c` / `jacobi_gpu.h`** — OpenMP target-offload GPU version.
  Grid stays device-resident for the whole run (`omp_target_alloc`,
  device-to-device `copy_grid`); only a single optional D→H copy at the end.
  Exits cleanly (code 0) if `omp_get_num_devices()==0`, so it doesn't break
  the pipeline on CPU-only nodes.

Build via `CMakeLists.txt` (C99, `-O3`, requires OpenMP + MPI). The GPU target
is a custom `nvc -mp=gpu -gpu=cc80` command, only added if `nvc` (NVHPC SDK,
`module load nvhpc`) is found in `PATH`; otherwise silently skipped.

## Benchmark pipeline

- **`main.py`** — CMake-builds everything, then runs the full parameter sweep
  *inside one SLURM job* (Leonardo's `lrd_all_serial` QOS caps queued jobs at
  2/user, so sub-job submission per config wasn't viable). Grid sizes
  `100–400`, OMP threads `1/2/4/8`, MPI procs `1/2/4/8` (kept ≤8 total workers
  for hybrid runs). Results append to a CSV via the `TIMING_CSV` env var
  (`constants_analysis.py`, default `timing_results_2.csv`). GPU runs are
  auto-skipped if `nvidia-smi` finds no device.
- **`submit_pipeline.sh`** — CPU-only submission, `lrd_all_serial` partition.
- **`submit_gpu_pipeline.sh`** — full CPU+GPU submission, `boost_usr_prod`
  partition with `--gres=gpu:1` (needed for real A100 runs).
- **`analysis.py`** — loads the CSV, produces `methods_comparison.png`
  (total time vs grid size, all methods/configs) and `speedup_vs_grid.png`
  (speedup over serial baseline), saved to `../Jacobi_plots/`.
- **`visualization_parallelization.ipynb`** — didactic notebook: thread
  ownership map, animated stencil-footprint over the grid, cache
  working-set analysis. Static outputs in `../Jacobi_plots/`
  (`thread_memory_map.png`, `collapse2_ownership.png`, `cache_working_set.png`),
  interactive versions in `../Jacobi_visualization/`
  (`cache_sliding_window.html`, `stencil_footprint.html`).

## Current status (last commit `69986c5` "final exam HPC", 2026-04-08)

- Pipeline ran successfully end-to-end on Leonardo (`lrd_all_serial`,
  CPU-only node) — see `../jacobi_pipeline_39290012.out`. `timing_results_2.csv`
  has 106 rows: `none` ×7, `just_omp` ×28, `omp_mpi` ×70, across grid sizes
  100–400.
- **GPU case is missing from the results** — that run's node had no GPU, so
  `jacobi_gpu` exited immediately (as designed). A separate submission of
  `submit_gpu_pipeline.sh` to `boost_usr_prod` is needed to populate `case=gpu`
  rows and the GPU line in the plots.
- **Folder reorg not yet committed.** `git status` shows the previous flat
  layout (`scripts/*.c`, `scripts/plots/`, `scripts/animations/`) deleted and
  the new layout (`scripts/Jacobi/`, `scripts/Jacobi_plots/`,
  `scripts/Jacobi_visualization/`) untracked. Needs `git add`/`commit` (ideally
  via `git mv` if redone, to keep file history attached).
- A stray zero-byte `CMake` file sits alongside `CMakeLists.txt` — looks like
  a leftover, not referenced by the build.

## Open items

- Run `submit_gpu_pipeline.sh` on an A100 node to get GPU numbers into
  `timing_results_2.csv` and the comparison plots.
- Commit the directory reorganisation.
- Decide whether the stray `CMake` file should be deleted.
