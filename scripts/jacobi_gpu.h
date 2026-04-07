#ifndef JACOBI_GPU_H
#define JACOBI_GPU_H

#include <math.h>
#include <omp.h>
#include <constants_jacobi.h>

/*
 * GPU parallelisation via OpenMP target offload (OpenMP 4.5+).
 * ------------------------------------------------------------
 * All kernel functions expect device-resident arrays allocated with
 * omp_target_alloc(); passing host pointers produces undefined behaviour.
 *
 * Design notes
 * ------------
 * is_device_ptr(...)
 *   Tells the runtime the pointers already live on the device (from
 *   omp_target_alloc); no implicit PCIe transfer is issued.  This is the
 *   correct idiom when device-native memory is used.
 *
 * fill_boundary_grid
 *   Two independent 1-D target loops set constant boundary values directly
 *   on the device — no host-to-device transfer for initialisation.
 *
 * fill_grid_withouth_boundary
 *   Collapsed 2-D loop: the full (nx × ny) interior is zeroed in a single
 *   kernel launch, giving maximum occupancy for a zero-work element.
 *
 * update
 *   collapse(2) exposes the full N = nx × ny work to the GPU's SIMT
 *   scheduler.  `teams distribute` assigns collapsed row×col chunks to
 *   thread-teams (CUDA blocks); `parallel for` assigns residual items to
 *   threads within each team (CUDA threads).  Consecutive j values in the
 *   inner loop map to consecutive memory addresses, so all threads in a
 *   warp access a contiguous cache line → coalesced loads and stores.
 *
 * check_convergence
 *   Collapsed 2-D loop with on-device reduction(+:tmpnorm).  OpenMP
 *   initialises the device-side private copy to 0, reduces across all
 *   threads, and writes the final value back to the host scalar before the
 *   target region exits — no explicit map clause needed for the scalar.
 *
 * copy_grid
 *   Device-to-device element-wise copy replacing the host memcpy call,
 *   avoiding a round-trip through PCIe on every Jacobi iteration.
 */

void fill_boundary_grid(double *grid, double *grid_new, int nx, int ny)
{
    const int ny2 = ny + 2;

    /* top (row 0) and bottom (row nx+1) */
    #pragma omp target teams distribute parallel for schedule(static) \
            is_device_ptr(grid, grid_new)
    for (int i = 0; i < ny2; i++) {
        grid[i]              = grid_new[i]              = TOP;
        grid[ny2*(nx+1) + i] = grid_new[ny2*(nx+1) + i] = BOTTOM;
    }

    /* left (col 0) and right (col ny+1) */
    #pragma omp target teams distribute parallel for schedule(static) \
            is_device_ptr(grid, grid_new)
    for (int i = 1; i <= nx; i++) {
        const int base = ny2 * i;
        grid[base]        = grid_new[base]        = LEFT;
        grid[base + ny+1] = grid_new[base + ny+1] = RIGHT;
    }
}

void fill_grid_withouth_boundary(double *grid, double *grid_new, int nx, int ny)
{
    #pragma omp target teams distribute parallel for collapse(2) schedule(static) \
            is_device_ptr(grid, grid_new)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            grid[i*(ny+2)+j] = grid_new[i*(ny+2)+j] = 0.0;
        }
    }
}

void update(double *grid, double *grid_new, int nx, int ny)
{
    /* collapse(2): distribute the full nx×ny work to the GPU thread hierarchy.
     * Consecutive j values → consecutive addresses → coalesced memory access. */
    #pragma omp target teams distribute parallel for collapse(2) schedule(static) \
            is_device_ptr(grid, grid_new)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            const int k = (ny+2)*i + j;
            grid_new[k] = 0.25 * (grid[k-1]      + grid[k+1]
                                + grid[k-(ny+2)] + grid[k+(ny+2)]);
        }
    }
}

double check_convergence(int nx, int ny, double *grid)
{
    double tmpnorm = 0.0;
    /* reduction(+:tmpnorm): OpenMP creates a device-private accumulator,
     * reduces across all teams/threads, and adds the result to the host
     * variable when the target region exits. */
    #pragma omp target teams distribute parallel for collapse(2) \
            schedule(static) reduction(+:tmpnorm) \
            is_device_ptr(grid)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            const int k = (ny+2)*i + j;
            const double r = grid[k]*4.0
                             - grid[k-1]      - grid[k+1]
                             - grid[k-(ny+2)] - grid[k+(ny+2)];
            tmpnorm += r * r;
        }
    }
    return sqrt(tmpnorm);
}

/* Device-to-device copy: replaces the host memcpy after each iteration,
 * keeping data on the GPU and avoiding a costly PCIe round-trip. */
void copy_grid(double *dst, double *src, int nx, int ny)
{
    const int N = (nx+2)*(ny+2);
    #pragma omp target teams distribute parallel for schedule(static) \
            is_device_ptr(dst, src)
    for (int k = 0; k < N; k++) {
        dst[k] = src[k];
    }
}

#endif /* JACOBI_GPU_H */
