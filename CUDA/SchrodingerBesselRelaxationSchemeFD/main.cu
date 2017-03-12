#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <string>

#define M_PI 3.141592653589793

#define NGrid 256
#define NBlock 512




#define N 131072
#define Nd 131072.

#define Xmax (1000.)
#define Xmin (-1000.)

#define L (Xmax - Xmin)

#define dx (L/(Nd-1.))

#define LAMBDA -1.
#define cmplx cuDoubleComplex


//Operator overloading for cmplx

__device__ __host__ __inline__ cmplx operator+(const cmplx &a, const cmplx &b)
{
	return cuCadd(a, b);
}

__device__ __host__ __inline__ cmplx operator+(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a + b.x, b.y);
}

__device__ __host__ __inline__ cmplx operator+(const cmplx  &a, const double &b)
{
	return make_cuDoubleComplex(a.x + b, a.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const cmplx &b)
{
	return cuCsub(a, b);
}

__device__ __host__ __inline__ cmplx operator-(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a - b.x, -b.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(a.x - b, a.y);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a)
{
	return make_cuDoubleComplex(-a.x, -a.y);
}

__device__ __host__ __inline__ cmplx operator*(const cmplx &a, const cmplx &b)
{
	return cuCmul(a, b);
}

__device__ __host__ __inline__ cmplx operator*(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}

__device__ __host__ __inline__ cmplx operator*(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(b*cuCreal(a), b*cuCimag(a));
}

__device__ __host__ __inline__ cmplx operator/(const cmplx &a, const cmplx &b)
{
	return cuCdiv(a, b);
}

__device__ __host__ __inline__ cmplx operator/(const cmplx &a, const double &b)
{
	return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__device__ __host__ __inline__ cmplx operator/(const double &a, const cmplx &b)
{
	return make_cuDoubleComplex(a, 0) / b;
}

__device__ __host__ __inline__ cmplx iMul(const cmplx &a)
{
	return make_cuDoubleComplex(-cuCimag(a), cuCreal(a));
}

__device__ __host__ __inline__ cmplx iMul(const double &a)
{
	return make_cuDoubleComplex(0, a);
}



//Init Pulse Kernel
__global__ void pulseBin(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	bool isLeft = i>(7 * N / 16);
	bool isRight = i<(9 * N / 16);
	if (i < N && isLeft && isRight)
	{
		V[i] = make_cuDoubleComplex(1., 0.);
	}

	//__syncthreads();
}

__global__ void pulseSin(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	V[i] = make_cuDoubleComplex(sin(Xmin + dx*static_cast<double>(i)), 0.);
}

__global__ void pulseXX(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	V[i] = make_cuDoubleComplex((static_cast<double>(i)*dx + Xmin)*(static_cast<double>(i)*dx + Xmin), 0);
}

__global__ void pulseGauss(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	double x = static_cast<double>(i)*dx + Xmin;
	if (i < N)
		V[i] = make_cuDoubleComplex(.5*std::exp(-x*x / 1.), 0);
}

//Init Phi, u must be initialized
__global__ void initPhi(cmplx* __restrict__ phi, cmplx* __restrict__ u)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		phi[i] = u[i] * cuConj(u[i]);

}

//Boundary manager
//Boundary Kernel
__global__ void boundKernel(cmplx *V)
{
	int i = threadIdx.x;
	if(i<2)
		V[i*(N - 1)] = make_cuDoubleComplex(0, 0);
}

__global__ void boundabsKernel(cmplx *V)
{//100 is the maximum of kernel requiere
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < 100)
	{
		V[i] = cuCmul(V[i], make_cuDoubleComplex(i / 100., 0));
		V[N - i - 1] = cuCmul(V[N - i - 1], make_cuDoubleComplex(i / 100., 0));
	}
}

void applyBoundaryCondition(cmplx *d_V)
{
	boundabsKernel << <1, 100 >> > (d_V);
	//boundKernel << <1, 2 >> > (d_V);
}



//Finite Difference
__device__ cmplx __inline__ derivative2ndFDCentral(const cmplx &x, const cmplx &xm, const cmplx &xp)
{
	//xm= x-1 , xp =x+1

	return (xm + xp + 2.*x) / dx / dx;
}

__device__ cmplx __inline__ derivative2ndFDLeft(const cmplx &x, const cmplx &xm, const cmplx &xmm, const cmplx &xmmm)
{
	//xmm= x-2 , xm =x-1

	return(2.*x - 5.*xm + 4.*xmm - xmmm) / dx / dx;
}

__device__ cmplx __inline__ derivative2ndFDRight(const cmplx &x, const cmplx &xp, const cmplx &xpp, const cmplx &xppp)
{
	//xpp= x+2 , xp =x+1

	return(2.*x - 5.*xp + 4.*xpp - xppp) / dx / dx;
}

__device__ cmplx derivative2nd(int i, cmplx *d_V)
{
	//d_V requiere at least 4 value

	if (i == 0)
		derivative2ndFDRight(d_V[i], d_V[i + 1], d_V[i + 2], d_V[i + 3]);
	if (i == N - 1)
		return derivative2ndFDLeft(d_V[i], d_V[i - 1], d_V[i - 2], d_V[i - 3]);
	else
		return derivative2ndFDCentral(d_V[i], d_V[i - 1], d_V[i + 1]);
}


//Compute the PHI at time n+1/2 with U at time t and PHI at time t-1/2
__global__ void computePhi(cmplx* __restrict__ phi, cmplx* __restrict__ u)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		phi[i] = 2.*u[i] * cuConj(u[i]) - phi[i];
}



__global__ void jacobiEvaluateB(cmplx* __restrict__ V, cmplx* __restrict__ phi, cmplx* __restrict__ B, double dt, double pdx)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		B[i] = V[i] * (iMul(1. / dt) + 1. / pdx / pdx + .5*LAMBDA*phi[i]) - (.5 / pdx / pdx)*(V[i + 1] + V[i - 1]);
}

__global__ void jacobiNLS(cmplx* __restrict__ V, cmplx* __restrict__ Vtmp, cmplx* __restrict__ B, cmplx* __restrict__ phi, int iter, double dt, double pdx)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	cmplx alphainv = 1. / (iMul(1. / dt) - 2. / pdx / pdx - .5*LAMBDA*phi[i]);
	cmplx beta = make_cuDoubleComplex(.5 / pdx / pdx, 0);

	for (int k = 0; k < iter; ++k)//while
	{
		cmplx sigma;
		//First pass
		if (i == 0)
		{
			sigma = beta*(0. + V[i + 1]);
		}
		else if (i == (N - 1))
		{
			sigma = beta*(V[i - 1] + 0.);
		}
		else
		{
			sigma = beta*(V[i - 1] + V[i + 1]);
		}
		Vtmp[i] = alphainv*(B[i] - sigma);

		if (i == 0)
			V[i] = make_cuDoubleComplex(0, 0);
		if (i == (N - 1))
			V[i] = make_cuDoubleComplex(0, 0);

		//Second pass

		if (i == 0)
		{
			sigma = beta*(0. + Vtmp[i + 1]);
		}
		else if (i == (N - 1))
		{
			sigma = beta*(Vtmp[i - 1] + 0.);
		}
		else
		{
			sigma = beta*(Vtmp[i - 1] + Vtmp[i + 1]);
		}
		V[i] = alphainv*(B[i] - sigma);

		if (i == 0)
			V[i] = make_cuDoubleComplex(0, 0);
		if (i == (N - 1))
			V[i] = make_cuDoubleComplex(0, 0);
	}
}

__global__ void computeError(cmplx* __restrict__ V, cmplx* __restrict__ B, cmplx* __restrict__ phi, int iter, double dt, double pdx)
{

}

//Write Data 
void writeInFile(cmplx *V, int fileX)
{
	std::ofstream file;
	file.open("data" + std::to_string(fileX) + ".ds");
	for (int i = 0; i < N; ++i)
	{
		double tmp = sqrt(V[i].x*V[i].x + V[i].y*V[i].y);
		//if (V[i].x < 0)
		//tmp = -tmp;
		file << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n";
	}
	file.close();

}

void writeInFile(cmplx *h_V, std::string S)
{
	std::ofstream file;
	file.open(S + ".ds");
	for (int i = 0; i < N; ++i)
	{
		double tmp = sqrt(h_V[i].x*h_V[i].x + h_V[i].y*h_V[i].y);
		if (h_V[i].x < 0)
			tmp = -tmp;
		file << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n";
	}
	file.close();
}

int main()
{
	cmplx *h_V;
	cmplx *d_V;
	cmplx *d_Vtmp;
	cmplx *d_B;
	cmplx *d_phi;

	//Allocate one array on the Host and one on the device
	//Use pinned memory on the host
	cudaMallocHost(&h_V, N * sizeof(cmplx));

	cudaMalloc(&d_V, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmp, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmpout, N * sizeof(cmplx));

	pulseGauss << <NGrid, NBlock >> > (d_V);
	std::cout << "dx :" << dx << std::endl;

	int Ni(100);
	int Nj(15000);



	double dt(0.0000001);

	std::cout << "Total Time : " << static_cast<double>((Ni*Nj))*dt << std::endl;

	for (int i = 0; i < Ni + 1; i++)
	{


		//Save the data in a file at each step
		cudaMemcpy(h_V, d_V, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		writeInFile(h_V, i);


		std::cout << "Time : " << static_cast<double>((i*Nj))*dt << std::endl;
		for (int j = 0; j < Nj + 1; ++j)
		{
			
		}

	}

	//writeInFile(h_VAllStep, d_VAllStep, Ni);

	cudaFreeHost(h_V);
	//cudaFreeHost(h_VAllStep);

	cudaFree(d_V);
	cudaFree(d_Vtmp);
	//cudaFree(d_VAllStep);

	return 0;
}