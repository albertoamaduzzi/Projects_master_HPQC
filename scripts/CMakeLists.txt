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

# OpenMP + MPI version (well-structured, using jacobi_omp_mpi.h)
add_executable(jacobi_omp_mpi jacobi_omp_mpi.c)
target_link_libraries(jacobi_omp_mpi m OpenMP::OpenMP_C MPI::MPI_C)

# GPU offload version (OpenMP target directives via nvc from NVHPC SDK)
# Built with a custom command so the main project compiler (gcc/mpicc) is
# untouched.  If nvc is not in PATH the target is simply absent and the
# rest of the build is unaffected.
#
# On Leonardo:  module load nvhpc   → puts nvc in PATH
# sm_80 = Ampere A100
option(ENABLE_GPU_OFFLOAD "Build jacobi_gpu with OpenMP target offload" ON)

if(ENABLE_GPU_OFFLOAD)
    find_program(NVC_COMPILER nvc DOC "NVIDIA HPC SDK C compiler (for GPU offload)")

    if(NVC_COMPILER)
        message(STATUS "Found nvc: ${NVC_COMPILER} – jacobi_gpu will target the A100 GPU")
        add_custom_command(
            OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/jacobi_gpu
            COMMAND ${NVC_COMPILER}
                        -mp=gpu -gpu=cc80 -O3
                        -I${CMAKE_CURRENT_SOURCE_DIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu.c
                        -o ${CMAKE_CURRENT_BINARY_DIR}/jacobi_gpu
                        -lm
                        -Wl,-z,noexecstack
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu.c
                    ${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/constants_jacobi.h
            COMMENT "Compiling jacobi_gpu with nvc -mp=gpu -gpu=cc80"
        )
        # Use a distinct target name to avoid a circular dependency with
        # the output file that shares the basename "jacobi_gpu".
        add_custom_target(jacobi_gpu_build ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/jacobi_gpu)
    else()
        message(WARNING
            "nvc not found – jacobi_gpu will NOT be built. "
            "Load the nvhpc module before configuring: 'module load nvhpc'")
    endif()
endif()
