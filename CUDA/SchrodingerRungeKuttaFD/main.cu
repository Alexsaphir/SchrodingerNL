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

#define NBlock 256
#define NThread 512




#define N 131072
#define Nd 131072.

#define Xmax (1000.)
#define Xmin (-1000.)

#define L (Xmax - Xmin)

#define dx (L/(Nd-1.))

#define cmplx cuDoubleComplex


//Operator overloading for cmplx

__device__ __host__ __inline__ cmplx operator+(const cmplx &a, const cmplx &b)
{
	return cuCadd(a, b);
}

__device__ __host__ __inline__ cmplx operator-(const cmplx &a, const cmplx &b)
{
	return cuCsub(a, b);
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

__device__ __host__ __inline__ cmplx iMul(const cmplx &a)
{
	return make_cuDoubleComplex(-cuCimag(a), cuCreal(a));
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
	//boundabsKernel << <1, 100 >> > (d_V);
	boundKernel << <1, 2 >> > (d_V);
}



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



__device__ cmplx evaluateF(double t, cmplx *d_V, int i)
{
	//f= i*(-.5*d²E/dx² + |E|²E)
	//return make_cuDoubleComplex(-std::exp(-t), 0);

	return iMul(1.*derivative2nd(i, d_V)- .5*cuConj(d_V[i])*d_V[i] * d_V[i]);
}



__global__ void rungeKutta2FirstStep(cmplx *V, cmplx *Vstep1,double t, double h)
{
	//Apply Runge Kutta 2 first step for each value of V
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
		Vstep1[i] = V[i] + (h *.5)*evaluateF(t, V , i);
}

__global__ void rungeKutta2SecondStep(cmplx *V, cmplx *Vstep1, double t, double h)
{
	//Apply Runge Kutta 2 second step for each value of V
	//tmp contains the value of the first step
	//And save the final result in V
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < N)
		V[i] = V[i] + h*evaluateF(t+h/2., Vstep1, i);
}

void rungeKutta2ODE(cmplx *d_V, cmplx *d_Vtmp,double t, double h)
{
	//Compute the derivative of d_V
	
	//Perform the first Step of RK2 ans save it in Vtmp
	rungeKutta2FirstStep << <NBlock, NThread >> > (d_V, d_Vtmp, t, h);
	rungeKutta2SecondStep << < NBlock, NThread >> > (d_V, d_Vtmp, t, h);
}



__global__ void rungeKutta4FirstStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k1 = evaluateF(t, V, i);
		d_VtmpOut[i] = V[i] + (h / 6.)*k1;
		Vstep[i] = V[i] + (h / 2.)*k1;
	}
}

__global__ void rungeKutta4SecondStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k2 = evaluateF(t + h / 2., Vstep, i);
		d_VtmpOut[i] = d_VtmpOut[i] + (h / 3.)*k2;
		Vstep[i] = V[i] + (h / 2.)*k2;
	}
}

__global__ void rungeKutta4ThirdStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k3 = evaluateF(t + h / 2., Vstep, i);
		d_VtmpOut[i] = d_VtmpOut[i] + (h / 3.)*k3;
		Vstep[i] = V[i] + (h / 2.)*k3;
	}
}

__global__ void rungeKutta4FourStep(cmplx *V, cmplx *Vstep, cmplx *d_VtmpOut, double t, double h)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	if (i < N)
	{
		cmplx k4 = evaluateF(t, Vstep, i);
		V[i] = d_VtmpOut[i] + (h / 6.)*k4;
	}
}

void rungeKutta4ODE(cmplx *d_V, cmplx *d_Vtmp, cmplx *d_VtmpOut, double t, double h)
{
	rungeKutta4FirstStep << <NBlock, NThread >> > (d_V, d_Vtmp, d_VtmpOut, t, h);
	rungeKutta4SecondStep << <NBlock, NThread >> > (d_V, d_Vtmp, d_VtmpOut, t, h);
	rungeKutta4ThirdStep << <NBlock, NThread >> > (d_V, d_Vtmp, d_VtmpOut, t, h);
	rungeKutta4FourStep << <NBlock, NThread >> > (d_V, d_Vtmp, d_VtmpOut, t, h);
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
		if (h_V[i].x < 0)
			tmp = -tmp;
		file << (static_cast<double>(i)*dx + Xmin) << " " << tmp << "\n";
	}
	file.close();
}

void writeInFile(double *h_V, double *d_V, int Nt)
{
	cudaMemcpy(h_V, d_V, N*(Nt+1) * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < Nt+1; ++i)
	{
		std::ofstream file;
		file.open("data" + std::to_string(i) + ".ds");
		for (int j = i*N; j < (N*i+N); ++j)
		{
			file << (static_cast<double>(j-i*N)*dx + Xmin) << " " << h_V[j] << "\n";
		}
		file.close();
	}
}

__global__ void SaveDataKernel(int t, cmplx *V, double *VAll)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	double tmp = sqrt(V[i].x*V[i].x + V[i].y*V[i].y);
	//if (V[i].x < 0)
		//tmp = -tmp;

	VAll[i + N*t] = tmp;

}

int main()
{
	cmplx *h_V;
	cmplx *d_V;
	cmplx *d_Vtmp;
	cmplx *d_Vtmpout;

	//Allocate one array on the Host and one on the device
	//Use pinned memory on the host
	cudaMallocHost(&h_V, N * sizeof(cmplx));
	
	cudaMalloc(&d_V, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmp, N * sizeof(cmplx));
	cudaMalloc(&d_Vtmpout, N * sizeof(cmplx));

	pulseGauss << <NBlock, NThread >> > (d_V);
	std::cout << "dx :" << dx << std::endl;

	int Ni(100);
	int Nj(15000);

	//Allocate a big array on host and device to save all time step
	//double *h_VAllStep;
	//double *d_VAllStep;
	//cudaMallocHost(&h_VAllStep, N*(Ni + 1) * sizeof(double));

	//cudaMalloc(&d_VAllStep, N*(Ni + 1) * sizeof(double));
	

	double dt(0.0000001);

	std::cout << "Total Time : " << static_cast<double>((Ni*Nj))*dt << std::endl;

	for (int i = 0; i < Ni+1; i++)
	{
		

		//Save the data in a file at each step
		cudaMemcpy(h_V, d_V, N * sizeof(cmplx), cudaMemcpyDeviceToHost);
		writeInFile(h_V, i);
		
		//SaveDataKernel << <NBlock, NThread >> > (i, d_V, d_VAllStep);

		std::cout << "Time : " << static_cast<double>((i*Nj))*dt << std::endl;
		for (int j = 0; j < Nj+1; ++j)
		{
			//rungeKutta2ODE(d_V, d_Vtmp,static_cast<double>((i*Nj+j))*dt, dt);
			rungeKutta4ODE(d_V, d_Vtmp, d_Vtmpout, static_cast<double>((i*Nj + j))*dt, dt);
			applyBoundaryCondition(d_V);
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