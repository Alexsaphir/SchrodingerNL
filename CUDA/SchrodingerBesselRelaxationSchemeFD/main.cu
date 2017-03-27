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

#define NBlock 131072/64
#define NThread 1024




#define N 134217728/512
#define Nd 134217728./512.

#define Xmax (100.)
#define Xmin (-100.)

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
		B[i] = V[i] * (iMul(1. / dt) + 1. / pdx / pdx/ 2. + .5*LAMBDA*phi[i]) - (.25 / pdx / pdx)*(V[i + 1] + V[i - 1]);
}


//We can use __constant__ for beta

__global__ void jacobiNLSPassOne(cmplx* __restrict__ V, cmplx* __restrict__ Vtmp, cmplx* __restrict__ B, cmplx* __restrict__ phi, double dt, double pdx)
{
	int i_global = blockIdx.x *blockDim.x + threadIdx.x;//True position in the array
	int i = threadIdx.x;//Position in the grid

	//copy a part of the array V in shared memory
	//We need NThread+2 cmplx 
	__shared__ cmplx Vs[NThread + 2];

	//Begin initialization of the shared memory
	Vs[i + 1] = V[i_global];
	if (i == 0 && i_global != 0)
	{
		//V[i_global-1] exist
		Vs[i] = V[i_global - 1];
	}

	if (i == NThread - 1 && i_global != N - 1)
	{
		//V[i_global+1] exist
		Vs[NThread + 1] = V[i_global + 1];
	}
	__syncthreads();

	if (i < N)
	{

		cmplx alphainv = 1. / (iMul(1. / dt) - .5 / pdx / pdx - .5*LAMBDA*phi[i_global]);
		cmplx beta = make_cuDoubleComplex(.25 / pdx / pdx, 0);
		cmplx sigma;
		//sigma = beta*(V[i - 1] + V[i + 1]);
		sigma = beta*(Vs[i] + Vs[i + 2]);

		if (i_global == 0)
		{
			//sigma = beta*(0. + V[i_global + 1]);
			sigma = beta*(0. + Vs[i + 2]);
		}
		else if (i_global == (N - 1))
		{
			//sigma = beta*(V[i - 1] + 0.);
			sigma = beta*(Vs[i] + 0.);
		}
		
		
		//Save the value of this iteration
		Vtmp[i_global] = alphainv*(B[i_global] - sigma);
		//Boundary Condition
		if (i_global == 0)
			Vtmp[i_global] = make_cuDoubleComplex(0, 0);
		if (i_global == (N - 1))
			Vtmp[i_global] = make_cuDoubleComplex(0, 0);
		__syncthreads();
	}
}


__global__ void jacobiNLSPassTwo(cmplx* __restrict__ V, cmplx* __restrict__ Vtmp, cmplx* __restrict__ B, cmplx* __restrict__ phi, double dt, double pdx)
{
	int i_global = blockIdx.x *blockDim.x + threadIdx.x;//True position in the array
	int i = threadIdx.x;//Position in the grid

						//copy a part of the array V in shared memory
						//We need NThread+2 cmplx 
	__shared__ cmplx Vs[NThread + 2];

	//Begin initialization of the shared memory
	//Copy current value of the thread
	Vs[i + 1] = Vtmp[i_global];

	//Copy Boundary of the array
	if (i == 0 && i_global != 0)
	{
		//V[i_global-1] exist
		Vs[i] = Vtmp[i_global - 1];
	}
	if (i == NThread - 1 && i_global != N - 1)
	{
		//V[i_global+1] exist
		Vs[NThread + 1] = Vtmp[i_global + 1];
	}
	__syncthreads();

	if (i < N)
	{

		cmplx alphainv = 1. / (iMul(1. / dt) - .5 / pdx / pdx - .5*LAMBDA*phi[i_global]);
		cmplx beta = make_cuDoubleComplex(.25 / pdx / pdx, 0);
		cmplx sigma;

		if (i_global == 0)
		{
			//sigma = beta*(0. + V[i_global + 1]);
			sigma = beta*(0. + Vs[i + 2]);
		}
		else if (i_global == (N - 1))
		{
			//sigma = beta*(V[i - 1] + 0.);
			sigma = beta*(Vs[i] + 0.);
		}
		else
		{
			//sigma = beta*(V[i - 1] + V[i + 1]);
			sigma = beta*(Vs[i] + Vs[i + 2]);
		}

		//Save the value of this iteration
		V[i_global] = alphainv*(B[i_global] - sigma);
		//Boundary Condition
		if (i_global == 0)
			V[i_global] = make_cuDoubleComplex(0, 0);
		if (i_global == (N - 1))
			V[i_global] = make_cuDoubleComplex(0, 0);
		__syncthreads();
	}
}

__global__ void computeError(cmplx* __restrict__ V, cmplx* __restrict__ B, cmplx* __restrict__ phi, int iter, double dt, double pdx)
{

}

void jacobiNLS(cmplx* __restrict__ V, cmplx* __restrict__ Vtmp, cmplx* __restrict__ B, cmplx* __restrict__ phi, double dt, double pdx, int nbiter)
{
	for (int k = 0; k < nbiter; ++k)
	{
		jacobiNLSPassOne << <NBlock, NThread >> > (V, Vtmp, B, phi, dt, pdx);
		jacobiNLSPassTwo << <NBlock, NThread >> > (V, Vtmp, B, phi, dt, pdx);
	}
}


//Write Data 
void writeInFile(cmplx *V, int fileX)
{
	std::ofstream file;
	file.open("data" + std::to_string(fileX) + ".ds");
	for (int i = 0; i < N; i+=2048)
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

void Write3DFile(cmplx *V, double t)
{
	std::ofstream file;
	file.open("data3D.ds", std::ios_base::app);
	for (int i = 0; i < N; i += 2048)
	{
		double tmp = sqrt(V[i].x*V[i].x + V[i].y*V[i].y);
		if (V[i].x < 0)
			tmp = -tmp;
		file << t << " " << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n\n";
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
	cudaMalloc(&d_B, N * sizeof(cmplx));
	cudaMalloc(&d_phi, N * sizeof(cmplx));

	pulseGauss << <NBlock, NThread >> > (d_V);
	initPhi << <NBlock, NThread >> > (d_phi, d_V);


	std::cout << "dx :" << dx <<std::endl <<cudaGetLastError()<< std::endl;

	int Ni(100);
	int Nj(10000);
	int ITER(100);//True number of iteration is ITER*2



	double dt(0.0001);

	std::cout << "Total Time : " << static_cast<double>((Ni*Nj))*dt << std::endl;

	for (int i = 0; i < Ni + 1; i++)
	{


		//Save the data in a file at each step
		std::cout << "Copy ...";
		cudaMemcpy(h_V, d_V, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		std::cout << "Write ...";
		//writeInFile(h_V, i);
		Write3DFile(h_V, static_cast<double>((i*Nj))*dt);
		std::cout << "End." << std::endl;
		


		std::cout << "Index :" << i <<" Time : " << static_cast<double>((i*Nj))*dt << std::endl;

		if (i == Ni-1)
			break;
		for (int j = 0; j < Nj + 1; ++j)
		{
			//applyBoundaryCondition(d_V);
			if (i != 0)
			if (!(i == 0 && j== 0))
				computePhi << <NBlock, NThread >> > (d_phi, d_V);
			
			jacobiEvaluateB << <NBlock, NThread >> > (d_V, d_phi, d_B, dt, dx);
			jacobiNLS(d_V, d_Vtmp, d_B, d_phi, dt, dx, ITER);
			cudaDeviceSynchronize();
		}

	}


	cudaFreeHost(h_V);
	

	cudaFree(d_V);
	cudaFree(d_Vtmp);
	cudaFree(d_B);
	cudaFree(d_phi);

	return 0;
}