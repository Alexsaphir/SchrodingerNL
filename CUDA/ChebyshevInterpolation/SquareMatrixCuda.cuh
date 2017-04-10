#if !defined SQUAREMATRIXCUDA_H
#define SQUAREMATRIXCUDA_H

class SquareMatrixCuda
{
	SquareMatrixCuda(int sizeMatrix);
	~SquareMatrixCuda();

private:
	int m_h_N;
//	cmplx *m_d_M;

};

#endif


