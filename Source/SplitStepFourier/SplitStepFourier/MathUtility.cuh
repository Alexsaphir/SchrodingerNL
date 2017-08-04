#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <vector>

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

namespace MathUtility
{
	void __device__ __host__ interpolationLagrange(double *x, cmplx *Res, int nbPtsX,double *xLR, cmplx *leftPts, cmplx *rightPts,int nbPtsLR);
	void __device__ BFSM1(cmplx * V, int N, double xmin, double xmax);
	void __device__ BFSM2(cmplx * V, int N, double xmin, double xmax);

}