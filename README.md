# Introduction
This repository contains the projects of High-Performance and Quantum Computing faced during the Master at the University of Bologna. Each Project has its own set of scripts inside the folder /scripts/name_project and relative documentation. Each project is self contained and not always complete. 
This repository is expliceteley thought to show the computational methods I have faced explicitely during this master. It is organized in macro areas as perceived by the 
author that acknowledges are somewhat arbitrary. 
I am still working on it to polish the scripts and make them publishable.
NumericalAnalysis: Inegrators
ParallelComputing: OpenMp, MPI, CUDA
QuantumComputing: Pulser, Pennylane, Qiskit
AI: Empty (TODO: Pytorch, Tensorflow)


## Numerical Analysis
1) Integration of simple ODE to benchmark different integrator using numba, jax, Pytorch and C++ to understand what kind of parallelism works better.
2) Using Gadget-4 replicate RANS simulation of urban canyon under isothermal and boundary conditions (Just report)
3) Shock-Wave scenario using Gadget-4, RAMSES (Just report)  

## Jacobi (MPI-OpenMP)
Solution of Laplace equation using OpenMP and MPI, showing how MPI performs much better in this task. This problem is particularly well suited for MPI kind of parallelization.

## Dilate (CUDA-OpenMP)
Realization of dilation kernel that changes the color of each pixel by the minimum found in a given stencil mask centered on the circle. This problem is particularly well suited for CUDA 
parellelization.

