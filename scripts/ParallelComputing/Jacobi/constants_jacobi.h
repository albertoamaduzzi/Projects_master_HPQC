// Grid boundary conditions
#define RIGHT 1.0
#define LEFT 1.0
#define TOP 1.0
#define BOTTOM 10.0

// Algorithm settings
#define TOLERANCE   0.0001
#define NPRINT      1000
#define MAX_ITER    500000
/* Convergence is checked only once every CHECK_EVERY iterations.
 * Each check costs one MPI_Allreduce (or an OMP reduction); skipping
 * most of them cuts synchronisation overhead by (CHECK_EVERY-1)/CHECK_EVERY
 * at the cost of up to CHECK_EVERY-1 extra Jacobi iterations. */
#define CHECK_EVERY 10
