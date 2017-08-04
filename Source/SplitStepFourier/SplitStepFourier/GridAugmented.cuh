#pragma once

#include "Axis.h"

#include "KernelUtility.cuh"
#include "ComplexCuda.cuh"

class GridAugmented
{
public:
	GridAugmented(double Xmin, double Xmax, int N, int N_padding);
	Axis getAxis() const;
	Axis getAxisData() const;

	cmplx* getHostData() const;
	cmplx* getDeviceData() const;

	cmplx* getHostPaddedData() const;
	cmplx* getDevicePaddedData() const;

	void syncDeviceToHost();
	void syncHostToDevice();


	~GridAugmented();

private:
	Axis m_X;
	Axis m_Xa;//Default axis
	int m_nbPts;
	int m_nbPtsPadding;

	

	//Useful data + padded at the end
	cmplx* m_h_V;
	cmplx* m_d_V;

	//Padded data beginnig
	cmplx *m_h_Vp;
	cmplx *m_d_Vp;
};

// m_nbPtsPadding: number of points use for the padding
// Number of points use for the usefull data: m_nbPts

