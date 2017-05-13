#include "RungeKutta.cuh"

namespace
{
	//Kernel 
	__global__ void RungeKuttaSumKernel(cmplx *Uf, cmplx *Af, double dt, double c, int N)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i < N)
		{
			Af[i] = Uf[i] + Af[i] * c*dt;
		}
	}

	__global__ void RungeKuttaIncResKernel(cmplx *Nextf, cmplx *Tf, double dt, double c, int N)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i < N)
		{
			Nextf[i] = Nextf[i] + Tf[i] * dt*c;
		}
	}

	__global__ void RungeKuttaInitResKernel(cmplx *Nextf, cmplx*Uf, int N)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i < N)
		{
			Nextf[i] = Uf[i];
		}
	}

	__global__ void evaluateFKernel(cmplx *d_xf, cmplx *d_yf, int N)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		double k(i);
		//Compute frequency 
		cmplx f;

		if (i < N / 2)
			f = iMul(k *2.*M_PI / static_cast<double>(N));//k*
		else if (i > N / 2)
			f = iMul((k - static_cast<double>(N))*2.*M_PI / static_cast<double>(N));//k*
		else
			f = iMul(0);//k=N/2
		__syncthreads();
		
		if (i < N)
		{
			d_yf[i] = d_xf[i]*(f*f);
		}
	}

	void RungeKuttaSum(cmplx *Uf, cmplx *Af, double dt, double c, int N)
	{
		RungeKuttaSumKernel << < KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (Uf, Af, dt, c, N);
	}

	void RungeKuttaIncRes(cmplx *Nextf, cmplx *Tf, double dt, double c, int N)
	{
		//Nextf+=Tf*dt*c
		RungeKuttaIncResKernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (Nextf, Tf, dt, c, N);
	}

	void RungeKuttaInitRes(cmplx *Nextf, cmplx *Uf, int N)
	{
		//Copy the value of Uf in Next
		RungeKuttaInitResKernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (Nextf, Uf, N);
	}
}

void evaluateF(cmplx *d_xf, cmplx *d_yf, int N)
{
	evaluateFKernel << <KernelUtility::computeNumberOfBlocks(1024, N), 1024 >> > (d_xf, d_yf, N);
}

void RungeKutta4(cmplx *d_Uf, cmplx *d_Nextf, cmplx *TmpA, cmplx *TmpB, double dt, int N)
{
	RungeKuttaInitRes(d_Nextf, d_Uf, N);

	//Compute k1
	evaluateF(d_Uf, TmpA, N);//Uf //TmpA
	RungeKuttaIncRes(d_Nextf, TmpA, dt, 1. / 6., N);

	//Compute k2
	RungeKuttaSum(d_Uf, TmpA, dt, .5, N);
	evaluateF(TmpA, TmpB, N);//TmpA //TmpB
	RungeKuttaIncRes(d_Nextf, TmpB, dt, 1. / 3., N);

	//Compute k3
	RungeKuttaSum(d_Uf, TmpB, dt, .5, N);
	evaluateF(TmpB, TmpA, N);//TmpB //TmpA
	RungeKuttaIncRes(d_Nextf, TmpA, dt, 1. / 3., N);

	//Compute k4
	RungeKuttaSum(d_Uf, TmpA, dt, 1., N);
	evaluateF(TmpA, TmpB, N);//TmpA //TmpB
	
	RungeKuttaIncRes(d_Nextf, TmpB, dt, 1. / 6., N);
}
