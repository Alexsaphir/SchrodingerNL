#include "matrix.h"

Matrix::Matrix(int row, int column): CoreMatrix(row , column)
{
	V.fill(cmplx(0.,0.),m_row*m_column);
}

cmplx Matrix::getValue(int i, int j) const
{
	return V.at(index(i,j));
}

cmplx Matrix::getValue(int i) const
{
	if (i<0 || i>=V.size())
		return cmplx(0,0);
	return V.at(i);
}

void Matrix::setValue(int i, int j, const cmplx &value)
{
	if (i<0 || j<0)
		return;
	if (i>=m_row || j>=m_column)
		return;
	V.replace(j + m_column*i, value);
}

void Matrix::setValue(int i, const cmplx &value)
{
	if (i<0 || i>=V.size())
		return;
	V.replace(i, value);
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
