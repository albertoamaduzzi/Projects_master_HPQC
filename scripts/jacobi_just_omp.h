#ifndef JACOBI_JUST_OMP_H
#define JACOBI_JUST_OMP_H

#include <math.h>
#include <constants_jacobi.h>
#include <omp.h>

/*
 * Parallelisation notes
 * ---------------------
 * Loop variables are declared C99-style inside each loop so they are
 * automatically private — no explicit private() clause needed.
 *
 * fill_boundary_grid   – now parallelised; previously serial.
 *
 * fill_grid_withouth_boundary / check_convergence
 *   – collapse(2): collapses both loops into one work-sharing space,
 *     giving finer-grained, better-balanced work items at no extra cost
 *     (no stencil reads, so no cross-thread cache-line conflicts).
 *
 * update – parallelised on the OUTER (row) loop only.
 *   Each thread gets a contiguous block of rows.  In row-major layout
 *   that block is contiguous in memory, so the vertical stencil reads
 *   grid[k-(ny+2)] and grid[k+(ny+2)] stay in L2/L3 cache as the inner
 *   loop advances.  Collapsing would scatter thread assignments across
 *   non-contiguous addresses and destroy this locality.
 *
 * schedule(static): stencil work is uniform per element; static gives
 *   minimum overhead and predictable, cache-friendly thread-to-row mapping.
 *
 * Why large grids are slower
 * --------------------------
 * When nx*ny exceeds L3 cache (~20-40 MB on current nodes), every
 * vertical-neighbour read grid[k±(ny+2)] misses cache and goes to DRAM.
 * This is intrinsic to the Jacobi stencil; it cannot be fixed by adding
 * more OpenMP threads (memory bandwidth saturates).  The MPI decomposition
 * is the correct remedy: each rank owns only local_nx rows, whose working
 * set fits in cache.
 */

void fill_boundary_grid(double *grid, double *grid_new, int nx, int ny)
{
    /* top (row 0) and bottom (row nx+1) — O(ny) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny + 2; i++) {
        grid_new[i]               = grid[i]               = TOP;
        grid_new[(ny+2)*(nx+1)+i] = grid[(ny+2)*(nx+1)+i] = BOTTOM;
    }
    /* left (col 0) and right (col ny+1) — O(nx) */
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= nx; i++) {
        int base = (ny+2)*i;
        grid_new[base]      = grid[base]      = LEFT;
        grid_new[base+ny+1] = grid[base+ny+1] = RIGHT;
    }
}

void fill_grid_withouth_boundary(double *grid, double *grid_new, int nx, int ny)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            int k = (ny+2)*i + j;
            grid_new[k] = grid[k] = 0.0;
        }
    }
}

void update(double *grid, double *grid_new, int nx, int ny)
{
    /* Outer-loop only: each thread owns contiguous rows → stencil neighbours
     * remain in cache as the inner loop sweeps left-to-right. */
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            int k = (ny+2)*i + j;
            grid_new[k] = 0.25 * (grid[k-1] + grid[k+1]
                                 + grid[k-(ny+2)] + grid[k+(ny+2)]);
        }
    }
}

double check_convergence(int nx, int ny, double *grid)
{
    double tmpnorm = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:tmpnorm)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            int k = (ny+2)*i + j;
            double r = grid[k]*4.0
                       - grid[k-1] - grid[k+1]
                       - grid[k-(ny+2)] - grid[k+(ny+2)];
            tmpnorm += r * r;
        }
    }
    return sqrt(tmpnorm);
}

#endif /* JACOBI_JUST_OMP_H */