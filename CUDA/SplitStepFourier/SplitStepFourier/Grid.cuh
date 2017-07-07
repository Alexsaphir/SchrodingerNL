#pragma once

#include "Axis.h"

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

class Grid
{
public:
	Grid(double Xmin, double Xmax, int N);


	Axis getAxis() const;

	cmplx* getHostData() const;
	cmplx* getDeviceData() const;

	void syncDeviceToHost();
	void syncHostToDevice();

	~Grid();
private:
	Axis m_X;
	int m_nbPts;

	cmplx* m_h_V;
	cmplx* m_d_V;

};