#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <constants_jacobi.h>
#include <jacobi_omp_mpi.h>

/* Append one CSV row; creates header if the file is empty/new.
 * Must be called ONLY by rank 0 to avoid concurrent file writes. */
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("usage: %s GRIDX GRIDY\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);

    int local_nx = nx / size;
    int remainder = nx % size;
    if (rank < remainder) {
        local_nx++;
    }

    /* OMP thread count is fixed in master thread before any parallel region */
    int omp_threads = omp_get_max_threads();

    if (rank == 0) {
        printf("grid size %d X %d  (MPI ranks: %d, OMP threads/rank: %d)\n",
               nx, ny, size, omp_threads);
    }

    double *grid     = (double*)calloc((local_nx + 2) * (ny + 2), sizeof(double));
    double *grid_new = (double*)calloc((local_nx + 2) * (ny + 2), sizeof(double));

    /* ---- per-rank timing accumulators ---- */
    double t_fill_boundary = 0.0, t_fill_interior = 0.0;
    double t_update = 0.0, t_convergence = 0.0;
    double t0;

    double t_total_start = MPI_Wtime();

    t0 = MPI_Wtime();
    fill_boundary_grid(grid, grid_new, local_nx, ny, rank, size);
    t_fill_boundary = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    fill_grid_withouth_boundary(grid, grid_new, local_nx, ny);
    t_fill_interior = MPI_Wtime() - t0;

    double bnorm = 0.0;
    t0 = MPI_Wtime();
    double local_bnorm = check_convergence_local(local_nx, ny, grid);
    t_convergence += MPI_Wtime() - t0;
    MPI_Allreduce(&local_bnorm, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bnorm = sqrt(bnorm);

    int iter;
    double norm = 0.0;
    double local_norm = 0.0;

    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Convergence: checked every CHECK_EVERY iterations, but never at
         * iter==0 (that would recompute bnorm itself, giving norm/bnorm=1).
         * norm retains its last computed value between checks.            */
        if (iter > 0 && iter % CHECK_EVERY == 0) {
            t0 = MPI_Wtime();
            local_norm = check_convergence_local(local_nx, ny, grid);
            t_convergence += MPI_Wtime() - t0;

            MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            norm = sqrt(norm) / bnorm;
            if (norm < TOLERANCE) break;
        }

        /* ---- Non-blocking halo exchange ---------------------------------- */
        MPI_Request reqs[4];
        int req_cnt = 0;

        if (rank > 0) {
            MPI_Isend(&grid[1         * (ny+2)], ny+2, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
            MPI_Irecv(&grid[0         * (ny+2)], ny+2, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        if (rank < size - 1) {
            MPI_Isend(&grid[local_nx  * (ny+2)], ny+2, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
            MPI_Irecv(&grid[(local_nx+1)*(ny+2)], ny+2, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }

        t0 = MPI_Wtime();
        /* ---- Update INTERIOR rows while halos are in flight -------------- *
         * Rows 2..local_nx-1 only need locally-owned neighbours; they do    *
         * NOT depend on the ghost rows being received.  We compute them now  *
         * so the MPI transfer and this work overlap.                         */
        if (local_nx > 2)
            update_range(grid, grid_new, ny, 2, local_nx - 1);

        /* Wait for ghost rows 0 and local_nx+1 to arrive */
        MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);

        /* ---- Update BOUNDARY rows (need received halos) ------------------ */
        update_range(grid, grid_new, ny, 1, 1);
        if (local_nx > 1)
            update_range(grid, grid_new, ny, local_nx, local_nx);
        t_update += MPI_Wtime() - t0;

        /* Copy only interior rows (1..local_nx), leaving ghost rows 0 and
         * local_nx+1 intact with the values received via the MPI exchange
         * above.  Overwriting them here with grid_new's stale zero-initialised
         * ghost rows would corrupt the residual computed by check_convergence
         * at the top of the next iteration (the measured norm would grow as
         * interior values become non-zero while ghost rows stay at 0). */
        memcpy(&grid[1*(ny+2)], &grid_new[1*(ny+2)],
               sizeof(double) * local_nx * (ny + 2));

        if (rank == 0 && iter % NPRINT == 0) {
            printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
        }
    }

    double t_total = MPI_Wtime() - t_total_start;

    if (rank == 0) {
        printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);
    }

    /* Reduce per-rank times to rank 0 using MAX (bottleneck perspective).
     * All timing variables are scalars in the master thread → no race conditions. */
    double g_fill_boundary, g_fill_interior, g_update, g_convergence, g_total;
    MPI_Reduce(&t_fill_boundary, &g_fill_boundary, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_fill_interior, &g_fill_interior, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_update,        &g_update,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_convergence,   &g_convergence,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total,         &g_total,         1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Only rank 0 writes to avoid concurrent file access */
    if (rank == 0) {
        const char *csv_path = getenv("TIMING_CSV");
        if (!csv_path) csv_path = "timing_results.csv";
        write_csv(csv_path, nx, ny, omp_threads, size, "omp_mpi",
                  g_fill_boundary, g_fill_interior,
                  g_update, g_convergence, g_total, iter);
    }

    free(grid);
    free(grid_new);
    MPI_Finalize();
    return 0;
}

