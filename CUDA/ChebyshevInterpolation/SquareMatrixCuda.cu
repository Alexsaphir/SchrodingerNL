#include "SquareMatrixCuda.cuh"

SquareMatrixCuda::SquareMatrixCuda(int sizeMatrix)
{
	if (sizeMatrix > 0)
		m_h_N = sizeMatrix;
	else
		m_h_N = 0;
}

SquareMatrixCuda::~SquareMatrixCuda()
{
}
