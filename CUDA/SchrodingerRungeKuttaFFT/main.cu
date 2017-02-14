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

#define NGrid 512
#define NBlock 512

#define N 262144
#define Nd 262144.

#define Xmax (50.)
#define Xmin (-50.)

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
		V[i] = make_cuDoubleComplex(2.*std::exp(-x*x / 4.), 0);
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

void rungeKutta4(cmplx *d_Vc, cmplx *d_Vn, cmplx *d_Vder, cmplx *d_Vtmp, cufftHandle &plan, double h)
{
	//Value at time n are in d_Vc
	//Value computed are saved in d_Vn
	//d_Vder save the derivative compute with the fft

	//f= i*(d²E/dx² + |E|²E)

	//fill d_Vn with the value of d_Vc
	fillKernel << <NGrid, NBlock >> > (d_Vn, d_Vc);


	//Compute vector A =f(d_Vc)
	//1-Compute 2nd derivative with the fft saved in d_Vder
	derivative2nd(d_Vc, d_Vder, plan);
	//2-Evaluate f
	evaluateFKernel << <NGrid, NBlock >> > (d_Vc, d_Vder, d_Vtmp);
	//Now Vtmp contains A
	//Update d_Vn with partial sum h/6*A
	incrementMul1Kernel << <NGrid, NBlock >> > (d_Vn, d_Vtmp, make_cuDoubleComplex(h / 6., 0));




	//Compute B=f(d_VC+h/2*A)
	//0-Compute the fiber and save the result in d_Vtmp(A)
	incrementMul0Kernel << <NGrid, NBlock >> > (d_Vtmp, d_Vc, make_cuDoubleComplex(h / 2., 0));
	//1-Compute the 2nd derivative
	derivative2nd(d_Vtmp, d_Vder, plan);
	//2-Evaluate f
	evaluateFKernel << <NGrid, NBlock >> > (d_Vtmp, d_Vder, d_Vtmp);
	//Now Vtmp contains B
	//Update d_Vn with partial sum h/3*A
	incrementMul1Kernel << <NGrid, NBlock >> > (d_Vn, d_Vtmp, make_cuDoubleComplex(h / 3., 0));




	//Compute C=f(d_VC+h/2*B)
	//0-Compute the fiber and save the result in d_Vtmp(B)
	incrementMul0Kernel << <NGrid, NBlock >> > (d_Vtmp, d_Vc, make_cuDoubleComplex(h / 2., 0));
	//1-Compute the 2nd derivative
	derivative2nd(d_Vtmp, d_Vder, plan);
	//2-Evaluate f
	evaluateFKernel << <NGrid, NBlock >> > (d_Vtmp, d_Vder, d_Vtmp);
	//Now Vtmp contains B
	//Update d_Vn with partial sum h/3*A
	incrementMul1Kernel << <NGrid, NBlock >> > (d_Vn, d_Vtmp, make_cuDoubleComplex(h / 3., 0));




	//Compute D=f(d_VC+h*C)
	//0-Compute the fiber and save the result in d_Vtmp(A)
	incrementMul0Kernel << <NGrid, NBlock >> > (d_Vtmp, d_Vc, make_cuDoubleComplex(h, 0));
	//1-Compute the 2nd derivative
	derivative2nd(d_Vtmp, d_Vder, plan);
	//2-Evaluate f
	evaluateFKernel << <NGrid, NBlock >> > (d_Vtmp, d_Vder, d_Vtmp);
	//Now Vtmp contains B
	//Update d_Vn with partial sum h/3*A
	incrementMul1Kernel << <NGrid, NBlock >> > (d_Vn, d_Vtmp, make_cuDoubleComplex(h / 6., 0));

	//
}

void rungeKutta2(cmplx *d_Vc, cmplx *d_Vn, cmplx *d_Vder, cmplx *d_Vtmp, cufftHandle &plan, double h)
{
	//fill d_Vn with the value of d_Vc
	fillKernel << <NGrid, NBlock >> > (d_Vn, d_Vc);


	//Compute vector A =f(d_Vc)
	//1-Compute 2nd derivative with the fft saved in d_Vder
	derivative2nd(d_Vc, d_Vder, plan);
	//2-Evaluate f
	evaluateFKernel << <NGrid, NBlock >> > (d_Vc, d_Vder, d_Vtmp);
	//Now Vtmp contains A
	
	//Compute B=f(d_Vc+h/2*dVtmp)

	incrementMul0Kernel << <NGrid, NBlock >> > (d_Vtmp, d_Vc, make_cuDoubleComplex(h / 2., 0));
	derivative2nd(d_Vtmp, d_Vder, plan);
	evaluateFKernel << <NGrid, NBlock >> > (d_Vtmp, d_Vder, d_Vtmp);
	
	incrementMul1Kernel << <NGrid, NBlock >> > (d_Vn, d_Vtmp, make_cuDoubleComplex(h, 0));

}


__device__ cmplx evaluateF(const cmplx &x, const cmplx &der2)
{

}

void rungeKutta2ODE(cmplx *d_Vc, cmplx *d_Vn, cmplx *d_Vder, cufftHandle &plan, double h)
{

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
			if (V[i].x < 0)
				tmp = -tmp;
		file << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n";
	}
	file.close();

}


int main()
{
	cmplx *h_Vc;
	cmplx *d_Vc;
	cmplx *d_Vn;
	cmplx *d_Vder;
	cmplx *d_Vtmp;

	//Allocate one array on the Host and one on the device
	//Use pinned memory on the host
	cudaMallocHost(&h_Vc, N * sizeof(cmplx));
	cudaMalloc(&d_Vc, N * sizeof(cmplx));
	cudaMalloc(&d_Vn, N * sizeof(cmplx));
	cudaMalloc(&d_Vder, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmp, N * sizeof(cmplx));

	pulseGauss << <NGrid, NBlock >> > (d_Vc);

	cufftHandle plan;
	//Create 1D FFT plan
	cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

	cudaMemcpy(h_Vc, d_Vc, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
	writeInFile(h_Vc, 0);

	for (int i = 0; i < 11; i++)
	{
		double dt(0.00001);
		applyBoundaryCondition(d_Vc);
		
		//Save the data in a file at each step
		cudaMemcpy(h_Vc, d_Vc, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		writeInFile(h_Vc, i);
		std::cout << "Time : " << dt*100.*static_cast<double>(i) << std::endl;
		for (int j = 0; j < 1; ++j)
		{
			applyBoundaryCondition(d_Vc);
			rungeKutta2(d_Vc, d_Vn, d_Vder, d_Vtmp, plan, dt);
			std::swap(d_Vc, d_Vn);
		}
	}


	cufftDestroy(plan);
	cudaFreeHost(h_Vc);
	cudaFree(d_Vc);
	cudaFree(d_Vn);
	cudaFree(d_Vder);
	cudaFree(d_Vtmp);


	return 0;
}