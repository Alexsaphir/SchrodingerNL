#pragma once

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

void SplitStep(cmplx * d_U, double dt, int N, double Length, cufftHandle *plan, int order);