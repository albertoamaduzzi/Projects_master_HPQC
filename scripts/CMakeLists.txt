cmake_minimum_required(VERSION 3.10)
project(JacobiHPC C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Serial version
add_executable(jacobi jacobi.c)
# librt provides clock_gettime on older glibc; on modern systems it is in libc
target_link_libraries(jacobi m rt)

# OpenMP version
add_executable(jacobi_just_omp jacobi_just_omp.c)
target_link_libraries(jacobi_just_omp m OpenMP::OpenMP_C)

# OpenMP + MPI version
add_executable(jacobi_omp_mpi jacobi_omp_mpi.c)
target_link_libraries(jacobi_omp_mpi m OpenMP::OpenMP_C MPI::MPI_C)
