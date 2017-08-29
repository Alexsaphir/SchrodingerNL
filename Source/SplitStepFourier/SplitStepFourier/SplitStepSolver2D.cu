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

void SplitStep2D(cmplx * d_U, double dt, int N, int Nx, int Ny, double Lx, double Ly, cufftHandle * plan, int order)
{
	/*switch (order)
	{
	case 2: phi2(d_U, dt, N, Lx, Ly, plan);
		break;
	case 4: phi4(d_U, dt, N, Lx, Ly, plan);
		break;
	default:phi1(d_U, dt, N, Lx, Ly, plan);
		break;
	}*/
}
