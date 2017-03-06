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

#define NGrid 128
#define NBlock 512

#define N 131072/2
#define Nd 131072./2.

#define Xmax (100.)
#define Xmin (-100.)

#define L (Xmax - Xmin)

#define dx (L/(Nd-1.))

#define cmplx cufftDoubleComplex


//Operator overloading for cmplx

__device__ __host__ __inline__ cmplx operator+(const cmplx &a, const cmplx &b)
{
	return cuCadd(a, b);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const cmplx &b)
{
	return cuCsub(a, b);
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
	return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}

__device__ __host__ __inline__ cmplx iMul(const cmplx &a)
{
	return make_cuDoubleComplex(-cuCimag(a), cuCreal(a));
}




__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
{
	double factor = exp(z.x);
	return make_cuDoubleComplex(factor * cos(z.y), factor * sin(z.y));
}

//Derivative Kernel
__global__ void derivativeFourierKernel(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	double k = static_cast<double>(i);
	cmplx Freq;


	//Positive Frequency are between 0 and <N/2
	//Negative Frequency are between N/2and <N
	if (i < N / 2)
		Freq = make_cuDoubleComplex(0, 2.*M_PI*k / L);
	else if (i > N / 2)
		Freq = make_cuDoubleComplex(0, 2.*M_PI*(k - Nd) / L);
	else
		Freq = make_cuDoubleComplex(0, 0);//k=N/2
	__syncthreads();
	if (i<N)
		V[i] = cuCmul(Freq, V[i]);
}

__global__ void derivative2ndFourierKernel(cmplx *V)
{
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

	Freq = cuCmul(Freq, Freq);
	if (i<N)
		V[i] = cuCmul(Freq, V[i]);
}

//Init Pulse Kernel
__global__ void pulseBin(cmplx *V)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	bool isLeft = i>(3 * N / 8);
	bool isRight = i<(5 * N / 8);
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
		V[i] = make_cuDoubleComplex(2.*std::exp(-x*x / 16.), 0);
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



void __global__ evaluateFKernel(cmplx *Vin, cmplx *Vder, cmplx *Vout)
{
	//f= i*(.5*d²E/dx² + |E|²E)

	int i = blockIdx.x *blockDim.x + threadIdx.x;
	cmplx i_cmplx = make_cuDoubleComplex(0., 1.);
	if (i < N)
		Vout[i] = cuCmul(i_cmplx, cuCadd(cuCdiv(Vder[i], make_cuDoubleComplex(2., 0)), cuCmul(Vin[i], cuCmul(Vin[i], cuConj(Vin[i])))));
}


void __global__ incrementKernel(cmplx *V, cmplx *V1)
{
	//V+=V1
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		V[i] = cuCadd(V[i], V1[i]);
}

void __global__ incrementMul1Kernel(cmplx *V, cmplx *V1, cmplx d)
{
	//V+=d*V1
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		V[i] = cuCadd(V[i], cuCmul(V1[i], d));
}

void __global__ incrementMul0Kernel(cmplx *V, cmplx *V1, cmplx d)
{
	//V=d*V+V1
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		V[i] = cuCadd(cuCmul(V[i], d), V1[i]);
}

void __global__ incrementMul01Kernel(cmplx *V, cmplx *V1, cmplx d, cmplx d1)
{
	//V=d*V+d1*V1
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		V[i] = cuCadd(cuCmul(V[i], d), cuCmul(V1[i], d1));
}

void __global__ fillKernel(cmplx *V, cmplx *V1)
{
	//V=V1
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		V[i] = V1[i];
}



void derivative2nd(cmplx *d_Vin, cmplx *d_Vout, cufftHandle &plan)
{
	cufftExecZ2Z(plan, d_Vin, d_Vout, CUFFT_FORWARD);
	derivative2ndFourierKernel << <NGrid, NBlock >> > (d_Vout);
	cufftExecZ2Z(plan, d_Vout, d_Vout, CUFFT_INVERSE);
	NormFFT << <NGrid, NBlock >> > (d_Vout);
}

__device__ cmplx evaluateF(double t, cmplx *d_V, cmplx *d_Vder, int i)
{
	//f= i*(-.5*d²E/dx² + |E|²E)
	//return make_cuDoubleComplex(-std::exp(-t), 0);

	return iMul(.5*d_Vder[i] - cuConj(d_V[i])*d_V[i] * d_V[i]);
}


__global__ void rungeKutta4FirstStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, cmplx *Vder, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k1 = evaluateF(t, V, Vder, i);
		d_VtmpOut[i] = V[i] + (h / 6.)*k1;
		Vstep[i] = V[i] + (h / 2.)*k1;
	}
}

__global__ void rungeKutta4SecondStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, cmplx *Vder, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k2 = evaluateF(t + h / 2., Vstep, Vder, i);
		d_VtmpOut[i] = d_VtmpOut[i] + (h / 3.)*k2;
		Vstep[i] = V[i] + (h / 2.)*k2;
	}
}

__global__ void rungeKutta4ThirdStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, cmplx *Vder, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k3 = evaluateF(t + h / 2., Vstep, Vder, i);
		d_VtmpOut[i] = d_VtmpOut[i] + (h / 3.)*k3;
		Vstep[i] = V[i] + (h / 2.)*k3;
	}
}

__global__ void rungeKutta4FourStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, cmplx *Vder, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k4 = evaluateF(t, Vstep, Vder, i);
		V[i] = d_VtmpOut[i] + (h / 6.)*k4;
	}
}

void rungeKutta4ODE(cmplx *d_V, cmplx *d_Vtmp, cmplx *d_VtmpOut, cmplx *Vder, double t, double h, cufftHandle &plan)
{
	derivative2nd(d_V, Vder, plan);
	rungeKutta4FirstStep << <NGrid, NBlock >> > (d_V, d_Vtmp, d_VtmpOut, Vder, t, h);
	derivative2nd(d_Vtmp, Vder, plan);
	rungeKutta4SecondStep << <NGrid, NBlock >> > (d_V, d_Vtmp, d_VtmpOut, Vder, t, h);
	derivative2nd(d_Vtmp, Vder, plan);
	rungeKutta4ThirdStep << <NGrid, NBlock >> > (d_V, d_Vtmp, d_VtmpOut, Vder, t, h);
	derivative2nd(d_Vtmp, Vder, plan);
	rungeKutta4FourStep << <NGrid, NBlock >> > (d_V, d_Vtmp, d_VtmpOut, Vder, t, h);
}



void disp(cmplx *V)
{
	for (int i = 0; i < N; ++i)
		std::cout << sqrt(V[i].x*V[i].x + V[i].y*V[i].y) << "\t";
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
		//if (h_V[i].x < 0)
			//tmp = -tmp;
		file << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n";
	}
	file.close();
}


int main()
{
	cmplx *h_V;
	cmplx *d_V;
	cmplx *d_Vder;
	cmplx *d_Vtmp;
	cmplx *d_Vtmpout;

	//Allocate one array on the Host and one on the device
	//Use pinned memory on the host
	cudaMallocHost(&h_V, N * sizeof(cmplx));
	cudaMalloc(&d_V, N * sizeof(cmplx));
	cudaMalloc(&d_Vder, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmp, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmpout, N * sizeof(cmplx));

	pulseGauss << <NGrid, NBlock >> > (d_V);

	cufftHandle plan;
	//Create 1D FFT plan
	cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

	int Ni(1);
	int Nj(10000000);

	//Allocate a big array on host and device to save all time step
	//double *h_VAllStep;
	//double *d_VAllStep;
	//cudaMallocHost(&h_VAllStep, N*(Ni + 1) * sizeof(double));

	//cudaMalloc(&d_VAllStep, N*(Ni + 1) * sizeof(double));


	double dt(0.00000001);

	std::cout << "Total Time : " << static_cast<double>((Ni*Nj))*dt << std::endl;

	for (int i = 0; i < Ni + 1; i++)
	{


		//Save the data in a file at each step
		cudaMemcpy(h_V, d_V, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		writeInFile(h_V, i);

		//SaveDataKernel << <NGrid, NBlock >> > (i, d_V, d_VAllStep);

		std::cout << "Time : " << static_cast<double>((i*Nj))*dt << std::endl;
		if (i == Ni)
			break;
		for (int j = 0; j < Nj + 1; ++j)
		{
			//rungeKutta2ODE(d_V, d_Vtmp,static_cast<double>((i*Nj+j))*dt, dt);
			rungeKutta4ODE(d_V, d_Vtmp, d_Vtmpout, d_Vder, static_cast<double>((i*Nj + j))*dt, dt,plan);
			applyBoundaryCondition(d_V);
		}

	}




	cufftDestroy(plan);
	cudaFreeHost(h_V);
	cudaFree(d_V);
	cudaFree(d_Vder);
	cudaFree(d_Vtmp);
	cudaFree(d_Vtmpout);


	return 0;
}