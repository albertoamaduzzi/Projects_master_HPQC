#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <constants_jacobi.h>
#include <jacobi_just_omp.h>

/* Append one CSV row; creates header if the file is empty/new.
 * Called only from the master thread – no race condition. */
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

int main(int argc, char*argv[]) {

    double bnorm, norm;

    if (argc !=3) {
        printf("usage: %s GRIDX GRIDY\n", argv[0]);
        return(1);
    }

    int nx=atoi(argv[1]);
    int ny=atoi(argv[2]);

    /* Number of OMP threads is fixed before any parallel region */
    int omp_threads = omp_get_max_threads();
    printf("grid size %d X %d  (OMP threads: %d)\n", nx, ny, omp_threads);

    /* ---- timing accumulators (master thread only) ---- */
    double t_fill_boundary = 0.0, t_fill_interior = 0.0;
    double t_update = 0.0, t_convergence = 0.0;
    double t0, t_total_start;

    // allocate grid and updated grid
    double *grid= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));
    double *grid_new= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));

    t_total_start = omp_get_wtime();

    // Initialise Grid boundaries  (serial – boundary is small)
    t0 = omp_get_wtime();
    fill_boundary_grid(grid, grid_new, nx, ny);
    t_fill_boundary = omp_get_wtime() - t0;

    // Initialise rest of grid
    t0 = omp_get_wtime();
    fill_grid_withouth_boundary(grid, grid_new, nx, ny);
    t_fill_interior = omp_get_wtime() - t0;

    // Compute initial norm
    t0 = omp_get_wtime();
    bnorm = check_convergence(nx, ny, grid);
    t_convergence += omp_get_wtime() - t0;

    // Main loop
    int iter;
    for (iter=0; iter<MAX_ITER; iter++) {

        // Convergence check – timing wraps the parallel region inside the function
        t0 = omp_get_wtime();
        norm = check_convergence(nx, ny, grid) / bnorm;
        t_convergence += omp_get_wtime() - t0;
        if (norm < TOLERANCE) break;

        // Update new grid
        t0 = omp_get_wtime();
        update(grid, grid_new, nx, ny);
        t_update += omp_get_wtime() - t0;

        // copy updated grid
        memcpy(grid, grid_new, sizeof(double) * (nx + 2) * (ny + 2));

        if (iter % NPRINT == 0) printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
    }

    double t_total = omp_get_wtime() - t_total_start;
    printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);

    /* ---- write timing CSV (master thread, outside parallel regions) ---- */
    const char *csv_path = getenv("TIMING_CSV");
    if (!csv_path) csv_path = "timing_results.csv";
    write_csv(csv_path, nx, ny, omp_threads, 1, "just_omp",
              t_fill_boundary, t_fill_interior,
              t_update, t_convergence, t_total, iter);

    // Do it at the end of all operations
    free(grid);
    free(grid_new);
    return 0;
}
