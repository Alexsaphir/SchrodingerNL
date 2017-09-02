#include "SplitStepSolver2D.cuh"

#define ALPHA 1./2.
#define BETA -1.
#define KAPA 2.

namespace
{
	__global__ void FFTResizekernel(cmplx * d_V, int Nx, int Ny)
	{
		int i = getGlobalIndex_2D_2D();

		int ix = blockDim.x*blockIdx.x + threadIdx.x;
		int iy = blockDim.y*blockIdx.y + threadIdx.y;

		if (ix < Nx && iy < Ny)
		{
			d_V[i] = cuCmul(d_V[i], make_cuDoubleComplex(1. / static_cast<double>(Nx*Ny), 0));
		}
	}

	void FFTResize(cmplx * d_V, int Nx, int Ny)
	{
		dim3 block;
		block.x = 32;
		block.y = 32;

		FFTResizekernel << <KernelUtility::computeNumberOfBlocks2D(block, Nx, Ny), block >> > (d_V, Nx, Ny);
	}
}

namespace
{
	__global__ void BoundaryConditionKernel(cmplx *V, int Nx, int Ny)
	{
		int i = blockIdx.x *blockDim.x + threadIdx.x;
		if (i == 0)
		{
			V[(Nx - 1)*Nx + (Ny - 1)] = V[0];
		}
		if (i >= 1 && i < Nx - 1)
		{
			V[(Nx - 1)*Nx + (i - 1)] = V[i - 1];
			V[(i - 1)*Nx + (Ny - 1)] = V[(i - 1)*Nx];
		}
	}

	void BoundaryCondition(cmplx *V, int Nx, int Ny)
	{
		//In 2d we have 2Nx+2(Ny-2) point for the boundary
		//Periodic Condition so 
		//we have only (Nx-1)+(Ny-2) points who keep their value
		//Point modify 1+(Nx-1)+(Ny-1)

		//(Nx-1,Ny-1):(0,0)
		//(Nx-1,0:Ny-2):(0,0:Ny-2)	bottom line
		//(0:Nx-2,Ny-1):(0:Nx-2,0)	Right column

		BoundaryConditionKernel << < KernelUtility::computeNumberOfBlocks(1024, Nx ), 1024 >> > (V, Nx, Ny);
	}
}

namespace
{
	__global__ void LinearKernel(cmplx *Vf, double dt, double ssm_p, int Nx, int Ny, double Lx, double Ly)
	{
		//ssm_p: coefficient of the split-step method

		int ix = blockDim.x*blockIdx.x + threadIdx.x;
		int iy = blockDim.y*blockIdx.y + threadIdx.y;

		//Compute value of k for each direction
		cmplx kx = make_cuDoubleComplex(0, 0);
		if (ix < Nx / 2)
		{
			kx = iMul(2.*M_PI *static_cast<double>(ix) / Lx);
		}
		if (ix == Nx / 2)
		{
			kx = make_cuDoubleComplex(0, 0);
		}
		if (ix > Nx / 2)
		{
			kx = iMul(2.*M_PI * (static_cast<double>(ix - Nx)) / Lx);
		}

		cmplx ky = make_cuDoubleComplex(0, 0);
		if (iy < Nx / 2)
		{
			ky = iMul(2.*M_PI *static_cast<double>(iy) / Ly);
		}
		if (iy == Ny / 2)
		{
			ky = make_cuDoubleComplex(0, 0);
		}
		if (iy > Ny / 2)
		{
			ky = iMul(2.*M_PI * (static_cast<double>(iy - Ny)) / Ly);
		}
		
			Vf[Nx*ix + iy] = cuCexp(iMul(ssm_p)*dt*ALPHA*kx*kx*ky*ky)*Vf[Nx*ix + iy];
	}

	__global__ void NLinearKernel(cmplx *V, double dt, double ssm_p, int Nx, int Ny, double Lx, double Ly)
	{
		int ix = blockDim.x*blockIdx.x + threadIdx.x;
		int iy = blockDim.y*blockIdx.y + threadIdx.y;
		
		if (ix < Nx && iy < Ny)
		{
			V[Nx*ix + iy] = cuCexp(iMul(dt)*ssm_p*BETA*KAPA*V[Nx*ix + iy] * cuConj(V[Nx*ix + iy]))*V[Nx*ix + iy];
		}
	}

	void LinearStep(cmplx *Vf, double dt, double ssm_p, int Nx, int Ny, double Lx, double Ly)
	{
		dim3 block;
		block.x = 32;
		block.y = 32;

		LinearKernel << <KernelUtility::computeNumberOfBlocks2D(block, Nx, Ny), block >> > (Vf, dt, ssm_p, Nx, Ny, Lx, Ly);
	}

	void NonLinearStep(cmplx *V, double dt, double ssm_p, int Nx, int Ny, double Lx, double Ly)
	{
		dim3 block;
		block.x = 32;
		block.y = 32;

		NLinearKernel << <KernelUtility::computeNumberOfBlocks2D(block, Nx, Ny), block >> > (V, dt, ssm_p, Nx, Ny, Lx, Ly);
	}
}

namespace
{
	void phi1(cmplx *d_U, double dt, int Nx, int Ny, double Lx, double Ly, cufftHandle *plan)
	{
		cufftExecZ2Z(*plan, d_U, d_U, CUFFT_FORWARD);
		LinearStep(d_U, dt, 1., Nx, Ny, Lx, Ly);
		cufftExecZ2Z(*plan, d_U, d_U, CUFFT_INVERSE);
		FFTResize(d_U, Nx, Ny);

		NonLinearStep(d_U, dt, 1., Nx, Ny, Lx, Ly);

	}
	void phi2(cmplx *d_U, double dt, int Nx, int Ny, double Lx, double Ly, cufftHandle *plan)
	{
		NonLinearStep(d_U, dt, .5, Nx, Ny, Lx, Ly);

		cufftExecZ2Z(*plan, d_U, d_U, CUFFT_FORWARD);
		LinearStep(d_U, dt, 1., Nx, Ny, Lx, Ly);
		cufftExecZ2Z(*plan, d_U, d_U, CUFFT_INVERSE);
		FFTResize(d_U, Nx, Ny);

		NonLinearStep(d_U, dt, .5, Nx, Ny, Lx, Ly);
	}

	void phi4(cmplx *d_U, double dt, int Nx, int Ny, double Lx, double Ly, cufftHandle *plan)
	{
		double w = (2. + std::pow(2., 1. / 3.) + .5*std::pow(2., 2. / 3)) / 3.;
		
		phi2(d_U, w*dt, Nx, Ny, Lx, Ly, plan);
		phi2(d_U, (1. - w)*dt, Nx, Ny, Lx, Ly, plan);
		phi2(d_U, w*dt, Nx, Ny, Lx, Ly, plan);
	}
}

void SplitStep2D(cmplx * d_U, double dt, int N, int Nx, int Ny, double Lx, double Ly, cufftHandle * plan, int order)
{
	switch (order)
	{
	case 2: phi2(d_U, dt, Nx, Ny, Lx, Ly, plan);
		break;
	case 4: phi4(d_U, dt, Nx, Ny, Lx, Ly, plan);
		break;
	default:phi1(d_U, dt, Nx, Ny, Lx, Ly, plan);
		break;
	}
}
