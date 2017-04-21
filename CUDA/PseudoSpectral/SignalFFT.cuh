#if !defined SIGNALFFT_H
#define SIGNALFFT_H

#include <new>
#include <cmath>
#define _USE_MATH_DEFINES

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "ComplexCuda.cuh"

#include "Signal.cuh"

class SignalFFT
{
public:
	SignalFFT(int NbPoints);
	
	int getSignalPoints() const;

	cmplx* getHostData() const;
	cmplx* getDeviceData() const;
	void syncHostToDevice();
	void syncDeviceToHost();

	void computeFFT(Signal *src);
	void ComputeSignal(Signal *dst);

//Filter only CPU data already sync on CPU, No sync perform after
	void smoothFilterCesaro();
	void smoothFilterLanczos();
	void smoothFilterRaisedCosinus();

	void reorderData();//All data are only modified on the CPU, No sync perform after
	void cancelReorderData();//Only CPU data, No sync perform after

	~SignalFFT();


private:
	int m_nbPts;
	cmplx *m_d_V;
	cmplx *m_h_V;
	cufftHandle m_plan;

	int m_block;
	int m_thread;

	bool m_GPUOrder;
};

__global__ void kernelResizeDataFFT(cmplx *d_V, int nbPts);
//__global__ void kernelSmoothing(cmplx *d_V,)

#endif