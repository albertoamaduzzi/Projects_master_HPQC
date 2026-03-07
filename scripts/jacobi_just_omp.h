#ifndef JACOBI_JUST_OMP_H
#define JACOBI_JUST_OMP_H

#include <math.h>
#include <constants_jacobi.h>
#include <omp.h>

// input bidimensional arrays 
void fill_boundary_grid(double *grid,          // array containing grid values (pass as reference) 
                        double *grid_new,      // array containing grid values (pass as reference)
                        int nx,                 // 
                        int ny                  // 
                        ){
    int i,j;
    for (i=0; i<ny+2; i++) {
        grid_new[i]=grid[i]=TOP;
        j=(ny+2)*(nx+1)+i;
        grid_new[j]=grid[j]=BOTTOM;
    }
    for (i=1; i<nx+1; i++) {
        j=(ny+2)*i;
        grid_new[j]=grid[j]=LEFT;
        grid_new[j+ny+1]=grid[j+ny+1]=RIGHT;
    }
}

void fill_grid_withouth_boundary(
                        double *grid,          // array containing grid values (pass as reference)
                        double *grid_new,      // array containing grid values (pass as reference)
                        int nx,                 // 
                        int ny                  //    
                        ){
    int i,j,k;
    // Initialise rest of grid
    #pragma omp parallel for private(i, j, k) shared(grid, grid_new, nx, ny)
    for (i=1; i<=nx; i++) {
        for (j=1; j<=ny; j++) {
            k=(ny+2)*i+j;
            grid_new[k]=grid[k]=0.0;
        }
    }
}


void update(double *grid,      // pass as reference
            double *grid_new, // pass as reference
            int nx,
            int ny
            ){
    int i,j,k;
    // Update new grid
    #pragma omp parallel for private(i, j, k) shared(grid, grid_new, nx, ny)
    for (i=1; i<=nx; i++) {
        for (j=1; j<=ny; j++) {
            k=(ny+2)*i+j;
            grid_new[k]=0.25 * (grid[k-1]+grid[k+1] + grid[k-(ny+2)] + grid[k+(ny+2)]);
        }
    }
}


double check_convergence(int nx,
                         int ny,
                         double *grid // do not pass as reference
                        ){
    // This is used to test convergence of the calculation
    double tmpnorm=0.0;
    int i, j, k;
    #pragma omp parallel for private(i, j, k) shared(grid, nx, ny) reduction(+:tmpnorm)
    for (i=1; i<=nx; i++) {
        for (j=1; j<=ny; j++) {
            k=(ny+2)*i+j;
            tmpnorm=tmpnorm+pow(grid[k]*4-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2);

        }
    }
    return sqrt(tmpnorm);
}

#endif