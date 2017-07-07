#include "SplitStepSolver.cuh"

namespace
{
	__global__ void FFTResizekernel(cmplx * d_V, int nbPts)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i < nbPts)
		{
			d_V[i] = cuCmul(d_V[i], make_cuDoubleComplex(1. / static_cast<double>(nbPts), 0));
		}
	}

	void FFTResize(cmplx * d_V, int N)
	{
		FFTResizekernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (d_V, N);
	}
}

namespace
{
	__global__ void LinearKernel(cmplx *Vf, double dt, double p, int N, double Length)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;


		if (i < N)
		{
			cmplx k = make_cuDoubleComplex(0, 0);
			if (i < N / 2)
			{
				k = iMul(2.*M_PI *static_cast<double>(i) / Length);
				//k = iMul(2.*M_PI *static_cast<double>(i) / static_cast<double>(N));
			}
			if (i == N / 2)
			{
				k = make_cuDoubleComplex(0, 0);
			}
			if (i > N / 2)
			{
				k = iMul(2.*M_PI * (static_cast<double>(i - N)) / Length);
				//k = iMul(2.*M_PI * (static_cast<double>(i - N)) / static_cast<double>(N));
			}
			Vf[i] = cuCexp(iMul(p)*dt*.5*k*k)*Vf[i];

		}
	}

	__global__ void NLinearKernel(cmplx *V, double dt, double p, int N, double Length)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		double kapa = 2.;
		if (i < N)
		{
			V[i] = cuCexp(iMul(dt)*p*kapa*V[i] * cuConj(V[i]))*V[i];
		}
	}

	void LinearStep(cmplx *Vf, double dt, double p, int N, double Length)
	{
		LinearKernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (Vf, dt, p, N, Length);
	}

	void NonLinearStep(cmplx *V, double dt, double p, int N, double Length)
	{
		NLinearKernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> >(V, dt, p, N, Length);
	}
}

void SplitStep(cmplx * d_U, double dt, int N, double Length, cufftHandle *plan)
{
	//Inplace computation
	
	//.5L+.5NL

	//FFt
	cufftExecZ2Z(*plan, d_U, d_U, CUFFT_FORWARD);
	//L
	LinearStep(d_U, dt, .5, N, Length);
	//FFt-1
	cufftExecZ2Z(*plan, d_U, d_U, CUFFT_INVERSE);
	FFTResize(d_U, N);
	//NL
	NonLinearStep(d_U, dt, .5, N, Length);

}