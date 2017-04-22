#if !defined SIGNAL_H
#define SIGNAL_H

#include <new>
#include <cmath>
#define _USE_MATH_DEFINES

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "ComplexCuda.cuh"

#include "Axis.h"


class Signal
{
public:
	Signal(double Xmin, double Xmax, int res);
	
	cmplx* getHostData() const;
	cmplx* getDeviceData() const;
	
	double getXmin() const;
	double getXmax() const;
	int getSignalPoints() const;
	void syncHostToDevice();
	void syncDeviceToHost();

	void fillHost(cmplx value);

	void fillDevice(cmplx value);

	void fillBoth(cmplx value);

	~Signal();

private:
	double m_Xmin;
	double m_Xmax;
	int m_nbPts;

	cmplx *m_h_V;
	cmplx *m_d_V;
};


void SinPulseEqui(Signal *S);
void CosWTPulseEqui(Signal *S);
void Trigo(Signal *S, double freq);
void GaussPulseLinear(Signal *S, double fc = 1000., double bw = .5, double bwr = -6., double tpr = -60);
void GaussCos(Signal *S);
#endif

