#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <constants_jacobi.h>
#include <jacobi.h>

/* Return wall-clock seconds using POSIX monotonic clock */
static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Append one CSV row; creates header if the file is empty/new */
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
    /* write header only when file is empty */
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

    printf("grid size %d X %d \n", nx, ny);

    /* ---- timing accumulators ---- */
    double t_fill_boundary = 0.0, t_fill_interior = 0.0;
    double t_update = 0.0, t_convergence = 0.0;
    double t0, t_total_start;

    // allocate grid and updated grid
    double *grid= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));
    double *grid_new= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));

    t_total_start = wtime();

    // Initialise Grid boundaries
    t0 = wtime();
    fill_boundary_grid(grid, grid_new, nx, ny);
    t_fill_boundary = wtime() - t0;

    // Initialise rest of grid
    t0 = wtime();
    fill_grid_withouth_boundary(grid, grid_new, nx, ny);
    t_fill_interior = wtime() - t0;

    // Compute initial norm
    t0 = wtime();
    bnorm = check_convergence(nx, ny, grid);
    t_convergence += wtime() - t0;

    // Main loop
    int iter;
    for (iter=0; iter<MAX_ITER; iter++) {

        // Convergence check
        t0 = wtime();
        norm = check_convergence(nx, ny, grid) / bnorm;
        t_convergence += wtime() - t0;
        if (norm < TOLERANCE) break;

        // Update new grid
        t0 = wtime();
        update(grid, grid_new, nx, ny);
        t_update += wtime() - t0;

        // copy updated grid
        memcpy(grid, grid_new, sizeof(double) * (nx + 2) * (ny + 2));

        if (iter % NPRINT == 0) printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
    }

    double t_total = wtime() - t_total_start;
    printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);

    /* ---- write timing CSV ---- */
    const char *csv_path = getenv("TIMING_CSV");
    if (!csv_path) csv_path = "timing_results.csv";
    write_csv(csv_path, nx, ny, 1, 1, "none",
              t_fill_boundary, t_fill_interior,
              t_update, t_convergence, t_total, iter);

    // Do it at the end of all operations
    free(grid);
    free(grid_new);
    return 0;
}
