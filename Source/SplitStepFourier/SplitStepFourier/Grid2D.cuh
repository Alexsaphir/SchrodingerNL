#pragma once

#include "Axis.h"

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

class Grid2D
{
public:
	Grid2D(double Xmin, double Xmax, int Nx, double Ymin, double Ymax, int Ny);


	Axis getAxisX() const;
	Axis getAxisY() const;

	cmplx* getHostData() const;
	cmplx* getDeviceData() const;

	void syncDeviceToHost();
	void syncHostToDevice();

	~Grid2D();
private:
	Axis m_X;
	Axis m_Y;
	int m_nbPts;
	int m_nbPtsX;
	int m_nbPtsY;

	cmplx* m_h_V;
	cmplx* m_d_V;

};

