#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// Grid boundary conditions
#define RIGHT 1.0
#define LEFT 1.0
#define TOP 1.0
#define BOTTOM 10.0

// Algorithm settings
#define TOLERANCE 0.0001
#define NPRINT 1000
#define MAX_ITER 500000

int main(int argc, char*argv[]) {
    int rank, size;
    int provided;
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("usage: %s GRIDX GRIDY\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);

    if (rank == 0) {
        printf("grid size %d X %d \n", nx, ny);
    }

    // 1D Domain Decomposition along rows (nx)
    int local_nx = nx / size;
    int remainder = nx % size;
    
    // Distribute remainder
    if (rank < remainder) {
        local_nx += 1;
    }

    // Allocate local grid and updated grid, including halos
    double *grid = (double*)malloc(sizeof(double) * (local_nx + 2) * (ny + 2));
    double *grid_new = (double*)malloc(sizeof(double) * (local_nx + 2) * (ny + 2));

    // Global row indices for this rank (to determine boundary conditions)
    int start_row;
    if (rank < remainder) {
        start_row = rank * local_nx;
    } else {
        start_row = rank * local_nx + remainder; // local_nx is smaller by 1 here
    }

    // Initialize boundaries
    // Top boundary (only for rank 0)
    if (rank == 0) {
        for (int j = 0; j < ny + 2; j++) {
            grid_new[j] = grid[j] = TOP;
        }
    }

    // Bottom boundary (only for rank size - 1)
    if (rank == size - 1) {
        for (int j = 0; j < ny + 2; j++) {
            int k = (local_nx + 1) * (ny + 2) + j;
            grid_new[k] = grid[k] = BOTTOM;
        }
    }

    // Left and Right boundaries for all ranks
    int i, j, k;
    for (i = 1; i <= local_nx; i++) {
        k = i * (ny + 2);
        grid_new[k] = grid[k] = LEFT;
        grid_new[k + ny + 1] = grid[k + ny + 1] = RIGHT;
    }

    // Initialize inner grid
    #pragma omp parallel for private(j, k)
    for (i = 1; i <= local_nx; i++) {
        for (j = 1; j <= ny; j++) {
            k = i * (ny + 2) + j;
            grid_new[k] = grid[k] = 0.0;
        }
    }

    double bnorm = 0.0;
    double local_bnorm_sq = 0.0;
    #pragma omp parallel for private(j, k) reduction(+:local_bnorm_sq)
    for (i = 1; i <= local_nx; i++) {
        for (j = 1; j <= ny; j++) {
            k = i * (ny + 2) + j;
            double val = grid[k] * 4 - grid[k - 1] - grid[k + 1] - grid[k - (ny + 2)] - grid[k + (ny + 2)];
            local_bnorm_sq += val * val;
        }
    }

    double global_bnorm_sq = 0.0;
    MPI_Allreduce(&local_bnorm_sq, &global_bnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bnorm = sqrt(global_bnorm_sq);

    double norm;
    int iter;
    int tag_down = 1;
    int tag_up = 2;

    for (iter = 0; iter < MAX_ITER; iter++) {
        // Compute convergence norm
        double local_tmpnorm_sq = 0.0;
        #pragma omp parallel for private(j, k) reduction(+:local_tmpnorm_sq)
        for (i = 1; i <= local_nx; i++) {
            for (j = 1; j <= ny; j++) {
                k = i * (ny + 2) + j;
                double val = grid[k] * 4 - grid[k - 1] - grid[k + 1] - grid[k - (ny + 2)] - grid[k + (ny + 2)];
                local_tmpnorm_sq += val * val;
            }
        }

        double global_tmpnorm_sq = 0.0;
        MPI_Allreduce(&local_tmpnorm_sq, &global_tmpnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        norm = sqrt(global_tmpnorm_sq) / bnorm;
        if (norm < TOLERANCE) break;

        // Halo Exchange
        MPI_Request reqs[4];
        int num_reqs = 0;

        // Send top row to rank-1, receive top halo from rank-1
        if (rank > 0) {
            MPI_Isend(&grid[1 * (ny + 2)], ny + 2, MPI_DOUBLE, rank - 1, tag_up, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(&grid[0 * (ny + 2)], ny + 2, MPI_DOUBLE, rank - 1, tag_down, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }

        // Send bottom row to rank+1, receive bottom halo from rank+1
        if (rank < size - 1) {
            MPI_Isend(&grid[local_nx * (ny + 2)], ny + 2, MPI_DOUBLE, rank + 1, tag_down, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(&grid[(local_nx + 1) * (ny + 2)], ny + 2, MPI_DOUBLE, rank + 1, tag_up, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }

        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

        // Update grid
        #pragma omp parallel for private(j, k)
        for (i = 1; i <= local_nx; i++) {
            for (j = 1; j <= ny; j++) {
                k = i * (ny + 2) + j;
                grid_new[k] = 0.25 * (grid[k - 1] + grid[k + 1] + grid[k - (ny + 2)] + grid[k + (ny + 2)]);
            }
        }

        // Copy grid
        // Need to copy ghosts? Or just inner? Inner is enough as halos will be exchanged
        #pragma omp parallel for private(j, k)
        for (i = 1; i <= local_nx; i++) {
            for (j = 1; j <= ny; j++) {
                k = i * (ny + 2) + j;
                grid[k] = grid_new[k];
            }
        }

        if (rank == 0 && iter % NPRINT == 0) {
            printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
        }
    }

    if (rank == 0) {
        printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);
    }

    free(grid);
    free(grid_new);

    MPI_Finalize();
    return 0;
}
