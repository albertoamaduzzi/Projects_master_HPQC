#ifndef JACOBI_OMP_MPI_H
#define JACOBI_OMP_MPI_H

#include <math.h>
#include <omp.h>

void fill_boundary_grid(double *grid, double *grid_new, int local_nx, int ny, int rank, int size) {
    #pragma omp parallel for
    for (int j = 0; j <= ny + 1; ++j) {
        if (rank == 0) {
            grid[0 * (ny + 2) + j] = 1.0;
            grid_new[0 * (ny + 2) + j] = 1.0;
        } else {
            grid[0 * (ny + 2) + j] = 0.0;
            grid_new[0 * (ny + 2) + j] = 0.0;
        }
        
        if (rank == size - 1) {
            grid[(local_nx + 1) * (ny + 2) + j] = 1.0;
            grid_new[(local_nx + 1) * (ny + 2) + j] = 1.0;
        } else {
            grid[(local_nx + 1) * (ny + 2) + j] = 0.0;
            grid_new[(local_nx + 1) * (ny + 2) + j] = 0.0;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i <= local_nx + 1; ++i) {
        grid[i * (ny + 2) + 0] = 1.0;
        grid[i * (ny + 2) + (ny + 1)] = 1.0;
        grid_new[i * (ny + 2) + 0] = 1.0;
        grid_new[i * (ny + 2) + (ny + 1)] = 1.0;
    }
}

void fill_grid_withouth_boundary(double *grid, double *grid_new, int local_nx, int ny) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= local_nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            grid[i * (ny + 2) + j] = 0.0;
            grid_new[i * (ny + 2) + j] = 0.0;
        }
    }
}

void update(double *grid, double *grid_new, int local_nx, int ny) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= local_nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            grid_new[i * (ny + 2) + j] = 0.25 * (
                grid[(i - 1) * (ny + 2) + j] + 
                grid[(i + 1) * (ny + 2) + j] + 
                grid[i * (ny + 2) + (j - 1)] + 
                grid[i * (ny + 2) + (j + 1)]
            );
        }
    }
}

double check_convergence(int local_nx, int ny, double *grid) {
    double temp = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:temp)
    for (int i = 1; i <= local_nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            double diff = 0.25 * (
                grid[(i - 1) * (ny + 2) + j] + 
                grid[(i + 1) * (ny + 2) + j] + 
                grid[i * (ny + 2) + (j - 1)] + 
                grid[i * (ny + 2) + (j + 1)]
            ) - grid[i * (ny + 2) + j];
            temp += diff * diff;
        }
    }
    return temp;
}

#endif // JACOBI_OMP_MPI_H
