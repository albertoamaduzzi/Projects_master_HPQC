#ifndef JACOBI_OMP_MPI_H
#define JACOBI_OMP_MPI_H

#include <math.h>
#include <omp.h>

/*
 * Design notes
 * ------------
 * fill_boundary_grid: the rank-dependent ghost-row value (0 or 1) is
 *   resolved once before the loop (compile-time-constant branch) so the
 *   parallel loop body is branch-free.  Two separate loops keep the two
 *   independent work items clean.
 *
 * update / update_range: the stencil is parallelised on the OUTER row loop
 *   only so that each thread's rows are contiguous in memory.  update_range
 *   allows the caller to update a subset of rows (used to overlap
 *   computation with MPI halo exchange — see jacobi_omp_mpi.c).
 *
 * collapse(2) + schedule(static) on fill/check: local_nx per rank is often
 *   small (nx/size rows), so collapsing both loops exposes more parallelism
 *   for a better-balanced thread distribution.
 */

void fill_boundary_grid(double *grid, double *grid_new,
                        int local_nx, int ny, int rank, int size)
{
    /* Resolve rank-dependent values once — no branching inside loops. */
    const double top_val    = (rank == 0)        ? 1.0 : 0.0;
    const double bottom_val = (rank == size - 1) ? 1.0 : 0.0;

    /* Ghost rows: top (row 0) and bottom (row local_nx+1) */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j <= ny + 1; ++j) {
        grid    [0             * (ny+2) + j] = top_val;
        grid_new[0             * (ny+2) + j] = top_val;
        grid    [(local_nx+1)  * (ny+2) + j] = bottom_val;
        grid_new[(local_nx+1)  * (ny+2) + j] = bottom_val;
    }
    /* Physical left/right boundaries (all ranks own these columns) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= local_nx + 1; ++i) {
        grid    [i*(ny+2) + 0]      = 1.0;
        grid    [i*(ny+2) + ny+1]   = 1.0;
        grid_new[i*(ny+2) + 0]      = 1.0;
        grid_new[i*(ny+2) + ny+1]   = 1.0;
    }
}

void fill_grid_withouth_boundary(double *grid, double *grid_new,
                                  int local_nx, int ny)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= local_nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            grid    [i*(ny+2)+j] = 0.0;
            grid_new[i*(ny+2)+j] = 0.0;
        }
    }
}

/* update_range: update rows [i_lo, i_hi] inclusive.
 * Used by jacobi_omp_mpi.c to split interior vs halo-dependent rows so
 * the interior update overlaps the non-blocking MPI halo exchange. */
void update_range(double *grid, double *grid_new,
                  int ny, int i_lo, int i_hi)
{
    #pragma omp parallel for schedule(static)
    for (int i = i_lo; i <= i_hi; ++i) {
        for (int j = 1; j <= ny; ++j) {
            grid_new[i*(ny+2)+j] = 0.25 * (
                grid[(i-1)*(ny+2)+j] + grid[(i+1)*(ny+2)+j] +
                grid[i*(ny+2)+(j-1)] + grid[i*(ny+2)+(j+1)]);
        }
    }
}

void update(double *grid, double *grid_new, int local_nx, int ny)
{
    update_range(grid, grid_new, ny, 1, local_nx);
}

double check_convergence(int local_nx, int ny, double *grid)
{
    double temp = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:temp)
    for (int i = 1; i <= local_nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            double diff = 0.25 * (
                grid[(i-1)*(ny+2)+j] + grid[(i+1)*(ny+2)+j] +
                grid[i*(ny+2)+(j-1)] + grid[i*(ny+2)+(j+1)]
            ) - grid[i*(ny+2)+j];
            temp += diff * diff;
        }
    }
    return temp;
}

#endif /* JACOBI_OMP_MPI_H */
