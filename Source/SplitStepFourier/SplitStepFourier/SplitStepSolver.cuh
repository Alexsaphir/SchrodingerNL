#pragma once


#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

void SplitStep(cmplx * d_U, double dt, int N, double Length, cufftHandle *plan, int order);

void SplitStep2D(cmplx *d_U, double dt, int N, int Nx, int Ny, double Lx, double Ly, cufftHandle *plan);
