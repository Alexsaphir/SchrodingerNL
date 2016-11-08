#include "matrix.h"

Matrix::Matrix(int row, int column): CoreMatrix(row , column)
{
	V.fill(cmplx(0.,0.),m_row*m_column);
}

cmplx Matrix::getValue(int i, int j) const
{
	return V.at(index(i,j));
}

void Matrix::setValue(int i, int j, const cmplx &value)
{
	if (i>=m_row || j>=m_column)
		return;
	V.replace(j + m_column*i, value);
}

int Matrix::index(int i, int j) const
{
	if (i>=m_row || j>=m_column)
		return 0;
	return j + m_column*i;
}

Matrix::~Matrix()
{
}
