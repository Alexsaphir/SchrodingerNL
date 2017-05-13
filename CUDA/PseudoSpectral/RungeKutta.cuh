#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "ComplexCuda.cuh"
#include "KernelUtility.cuh"


void RungeKutta4(cmplx * d_Uf, cmplx * d_Nextf, cmplx * TmpA, cmplx * TmpB, double dt, int N);