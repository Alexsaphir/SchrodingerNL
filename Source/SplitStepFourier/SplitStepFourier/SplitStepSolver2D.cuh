#pragma once

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

void SplitStep2D(cmplx *d_U, double dt, int N, int Nx, int Ny, double Lx, double Ly, cufftHandle *plan, int order);