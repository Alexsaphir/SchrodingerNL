#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <string>

#define M_PI 3.141592653589793

#define NBlock 2048
#define NThread 512

#define N 1048576
#define Nd 1048576.

#define Xmax (200.)
#define Xmin (-200.)

#define L (Xmax - Xmin)

#define dx (L/(Nd-1.))

#define cmplx cufftDoubleComplex


__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
{
	double factor = exp(z.x);
	return make_cuDoubleComplex(factor * cos(z.y), factor * sin(z.y));
}





//Init Pulse Kernel
__global__ void pulseBin(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	bool isLeft = i>(3*N/8);
	bool isRight= i<(5*N/8);
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
	if (i<N)
		V[i] = make_cuDoubleComplex(2.*std::exp(-x*x / 25.), 0);
}

//Normalize After fft Kernel
__global__ void NormFFT(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	cuDoubleComplex factor = make_cuDoubleComplex(N, 0);

	V[i] = cuCdiv(V[i], factor);
}

//Boundary manager
//Boundary Kernel
__global__ void boundKernel(cmplx *V)
{
	int i = threadIdx.x;
	V[i*(N - 1)] = make_cuDoubleComplex(0, 0);
}

__global__ void boundabsKernel(cmplx *V)
{//100 is the maximum of kernel requiere
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	V[i] = cuCmul(V[i], make_cuDoubleComplex(i / 100., 0));
	V[N - i - 1] = cuCmul(V[N - i - 1], make_cuDoubleComplex(i / 100., 0));
}

void applyBoundaryCondition(cmplx *d_V)
{
	boundabsKernel << <1, 100 >> > (d_V);
	//boundKernel << <1, 2 >> > (d_V);
}


//Linear Kernel Step
__global__ void LinearStepKernel(cmplx *V, double dt)
{
	//the data must be in fourrier space.
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	cmplx Freq;

	//Positive Frequency are between 0 and <N/2
	//Negative Frequency are between N/2and <N
	if (i < N / 2)
		Freq = make_cuDoubleComplex(0, 2.*M_PI*static_cast<double>(i) / L);
	else if (i > N / 2)
		Freq = make_cuDoubleComplex(0, 2.*M_PI*(static_cast<double>(i) - Nd) / L);
	else
		Freq = make_cuDoubleComplex(0, 0);//k=N/2
	__syncthreads();

	Freq = cuCmul(Freq, Freq);//k^2
	if(i<N)
		//V[i] = cuCmul(cuCexp(cuCmul(make_cuDoubleComplex(0, dt), Freq)), V[i]);
		V[i] = cuCmul(cuCexp(cuCmul(make_cuDoubleComplex(0, dt / 2.),Freq)), V[i]);
}
//Non Linear Kernel Step
__global__ void NonLinearStepKernel(cmplx *V, double dt)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i<N)
		V[i] = cuCmul(cuCexp(cuCmul(make_cuDoubleComplex(0, dt), cuCmul(V[i], cuConj(V[i])))), V[i]);
}

//Linear Step call  function
void linearStep(cmplx *d_V, double dt ,cufftHandle &plan)
{
	//plan must be already initialize
	cufftExecZ2Z(plan, d_V, d_V, CUFFT_FORWARD);
	LinearStepKernel << <NBlock, NThread >> > (d_V, dt);
	cufftExecZ2Z(plan, d_V, d_V, CUFFT_INVERSE);
	NormFFT << <NBlock, NThread >> > (d_V);
	applyBoundaryCondition(d_V);
}

void nonLinearStep(cmplx *d_V, double dt)
{
	NonLinearStepKernel << <NBlock, NThread >> > (d_V, dt);
	applyBoundaryCondition(d_V);
}




void disp(cmplx *V)
{
	for (int i = 0; i < N; ++i)
		std::cout <<sqrt(V[i].x*V[i].x+ V[i].y*V[i].y)<< "\t";
	std::cout << std::endl;
}

void dispError(cmplx *V_src, cmplx *V)
{
	double error(0);
	for (int i = 0; i < N; ++i)
	{
		double e = cuCabs(cuCsub(V[i], V_src[i]));
		if (e > error)
			error = e;
	}
	std::cout << "Error Max :" << error << std::endl;
}

void writeInFile(cmplx *V,int fileX)
{
	std::ofstream file;
	file.open("data" + std::to_string(fileX) + ".ds");
	for (int i = 0; i < N; ++i)
		file << (static_cast<double>(i)*dx + Xmin) << " " << sqrt(V[i].x*V[i].x + V[i].y*V[i].y) << "\n";
	file.close();

}

void approx2(cmplx *d_V, double dt, cufftHandle &plan)
{
	applyBoundaryCondition(d_V);
	nonLinearStep(d_V, dt / 2.);
	linearStep(d_V, dt, plan);
	nonLinearStep(d_V, dt / 2.);
}

int main()
{
	cmplx *h_V;
	cmplx *d_V;
	
	//Allocate one array on the Host and one on the device
	//Use pinned memory on the host
	cudaMallocHost(&h_V, N * sizeof(cmplx));
	cudaMalloc(&d_V, N * sizeof(cmplx));

	pulseGauss << <NBlock,NThread>> > (d_V);

	cufftHandle plan;
	//Create 1D FFT plan
	cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
	
	std::cout << "Space dx :" << dx << std::endl;
	
		
		for (int i = 0; i < 20001; i++)
	{
		double dt(0.001);
		applyBoundaryCondition(d_V);
		//Save the data in a file at each step
		cudaMemcpy(h_V, d_V, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		writeInFile(h_V, i);
		std::cout << "Time : " << dt*100.*static_cast<double>(i) << std::endl;
		

		double w = (2. + std::pow(2., 1. / 3.) + std::pow(2., -1. / 3.)) / 3.;

		for (int j = 0; j < 500; ++j)
		{
			applyBoundaryCondition(d_V);
			approx2(d_V, w*dt, plan);
			approx2(d_V, dt*(1. - 2.*w), plan);
			approx2(d_V, w*dt, plan);

		}
	}

	cufftDestroy(plan);
	cudaFreeHost(h_V);
	cudaFree(d_V);


	return 0;
}