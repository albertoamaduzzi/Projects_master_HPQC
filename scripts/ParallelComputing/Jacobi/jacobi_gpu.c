#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <constants_jacobi.h>
#include <jacobi_gpu.h>

/*
 * Memory strategy
 * ---------------
 * All grid data lives on the GPU for the entire run:
 *   - omp_target_alloc  allocates device-resident buffers.
 *   - fill_boundary_grid / fill_grid_withouth_boundary initialise them
 *     directly on the device — no host-to-device transfer for setup.
 *   - copy_grid (device-to-device) replaces memcpy after each iteration.
 *   - The only host-device transfer is a single device-to-host copy at
 *     the end (optional, for validation), timed as t_transfer.
 *
 * Timing
 * ------
 * omp_get_wtime() is used throughout consistent with the OMP versions.
 * Every omp target region is synchronous (no nowait), so the wall-clock
 * time around each call accurately measures GPU kernel execution.
 * t_transfer captures the cost of the final D→H copy so it can be
 * compared against the compute time.
 */

/* Append one CSV row; creates header if the file is empty/new. */
static void write_csv(const char *path,
                      int nx, int ny,
                      int omp_threads, int mpi_procs,
                      const char *cas,
                      double t_fill_boundary,
                      double t_fill_interior,
                      double t_update,
                      double t_convergence,
                      double t_total,
                      int iterations)
{
    FILE *f = fopen(path, "a+");
    if (!f) { fprintf(stderr, "Cannot open CSV %s\n", path); return; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "case,nx,ny,number_omp_threads,number_mpi_processes,"
                   "fill_boundary_grid_time,fill_grid_without_boundary_time,"
                   "update_time,check_convergence_time,total_time,iterations\n");
    }
    fprintf(f, "%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            cas, nx, ny, omp_threads, mpi_procs,
            t_fill_boundary, t_fill_interior,
            t_update, t_convergence, t_total, iterations);
    fclose(f);
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("usage: %s GRIDX GRIDY\n", argv[0]);
        return 1;
    }

    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    const size_t N = (size_t)(nx + 2) * (ny + 2);

    /* Query device availability */
    const int dev  = omp_get_default_device();
    const int host = omp_get_initial_device();
    const int num_devices = omp_get_num_devices();

    if (num_devices == 0) {
        fprintf(stderr,
                "[jacobi_gpu] No OpenMP target devices found on this node. "
                "The binary was compiled with GPU offload support but the "
                "current node has no GPU (e.g. lrd_all_serial partition). "
                "Submit to boost_usr_prod with --gres=gpu:1 for real A100 runs.\n");
        /* Exit 0 so the pipeline is not aborted when running on CPU-only nodes. */
        return 0;
    }

    printf("grid size %d X %d  (GPU device: %d / %d devices)\n",
           nx, ny, dev, num_devices);

    /* ---- Allocate device-resident buffers ---- */
    double *d_grid     = omp_target_alloc(N * sizeof(double), dev);
    double *d_grid_new = omp_target_alloc(N * sizeof(double), dev);

    if (!d_grid || !d_grid_new) {
        fprintf(stderr, "[jacobi_gpu] omp_target_alloc failed (N=%zu)\n", N);
        return 1;
    }

    /* ---- Timing accumulators ---- */
    double t_fill_boundary = 0.0, t_fill_interior = 0.0;
    double t_update = 0.0, t_convergence = 0.0;
    double t0;

    const double t_total_start = omp_get_wtime();

    /* ---- Initialise boundaries directly on device ---- */
    t0 = omp_get_wtime();
    fill_boundary_grid(d_grid, d_grid_new, nx, ny);
    t_fill_boundary = omp_get_wtime() - t0;

    /* ---- Initialise interior directly on device ---- */
    t0 = omp_get_wtime();
    fill_grid_withouth_boundary(d_grid, d_grid_new, nx, ny);
    t_fill_interior = omp_get_wtime() - t0;

    /* ---- Initial norm (baseline for relative convergence) ---- */
    t0 = omp_get_wtime();
    double bnorm = check_convergence(nx, ny, d_grid);
    t_convergence += omp_get_wtime() - t0;

    /* ---- Main Jacobi loop ---- */
    int iter;
    double norm = 0.0;

    for (iter = 0; iter < MAX_ITER; iter++) {

        t0 = omp_get_wtime();
        norm = check_convergence(nx, ny, d_grid) / bnorm;
        t_convergence += omp_get_wtime() - t0;

        if (norm < TOLERANCE) break;

        t0 = omp_get_wtime();
        update(d_grid, d_grid_new, nx, ny);
        t_update += omp_get_wtime() - t0;

        /* Device-to-device copy (no PCIe round-trip) */
        copy_grid(d_grid, d_grid_new, nx, ny);

        if (iter % NPRINT == 0)
            printf("Iteration=%d, Relative norm=%e\n", iter, norm);
    }

    const double t_total = omp_get_wtime() - t_total_start;

    printf("Terminated on %d iterations, Relative Norm=%e\n", iter, norm);

    /* ---- Optional: time the final device-to-host transfer ---- */
    double *h_grid = malloc(N * sizeof(double));
    if (h_grid) {
        const double t_d2h_start = omp_get_wtime();
        omp_target_memcpy(h_grid, d_grid,
                          N * sizeof(double), 0, 0, host, dev);
        printf("Device-to-host transfer: %.6f s\n",
               omp_get_wtime() - t_d2h_start);
        free(h_grid);
    }

    /* ---- Free device memory ---- */
    omp_target_free(d_grid,     dev);
    omp_target_free(d_grid_new, dev);

    /* ---- Write CSV (1 GPU = 1 process, no host OMP threads) ---- */
    const char *csv_path = getenv("TIMING_CSV");
    if (!csv_path) csv_path = "timing_results.csv";
    write_csv(csv_path, nx, ny,
              /*omp_threads=*/1, /*mpi_procs=*/1, "gpu",
              t_fill_boundary, t_fill_interior,
              t_update, t_convergence, t_total, iter);

    return 0;
}
