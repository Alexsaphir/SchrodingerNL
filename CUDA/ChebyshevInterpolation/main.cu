#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <string>

#include "ComplexCuda.cuh"

#define M_PI 3.1415926535897932384626433832795028841971693993751
//2*M_PI
#define M_PI2 6.2831853071795864769252867665590057683943387987502


#define N 32

#define NBlock 1
#define NThread 32



__global__ void computeChebyshevPoints(int nbPoints, double *d_V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < nbPoints)
		d_V[i] = -std::cos(static_cast<double>(i)*M_PI / static_cast<double>(nbPoints - 1));
	if (i < nbPoints)
		printf("%f  ", d_V[i]);
}

void computeChebyshevFirstDerivative(cmplx *d_Vin, cmplx *d_Vout, cmplx *d_Vtmp, int nbPoints)
{
	//Init V with the value of v
	initChebyshevDerivative << <NBlock, NThread>> > (d_Vin, d_Vout, nbPoints);


}

__global__ void initChebyshevDerivative(cmplx *d_Vin, cmplx *d_Vout, int nbPoints)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < nbPoints)
		d_Vout[2 * nbPoints - i] = d_Vin[i];
}


int main()
{
	cmplx *d_V;
	cmplx *d_Vtmp;

	int NbPoints(16);
	cudaMalloc(&d_V, NbPoints * sizeof(cmplx));
	cudaMalloc(&d_Vtmp, 2 * NbPoints * sizeof(cmplx));
	
	
	//computeChebyshevPoints << <1, N >> > (NbPoints, d_V);
	cudaDeviceSynchronize();
	getchar();

	return 0;
}
