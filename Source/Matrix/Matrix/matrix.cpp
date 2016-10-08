#include "matrix.h"

Matrix::Matrix(uint row, uint column): CoreMatrix(row , column)
{
	V.fill(cmplx(0.,0.),m_row*m_column);
}

cmplx Matrix::getValue(uint i, uint j) const
{
	return V.at(index(i,j));
}

void Matrix::setValue(uint i, uint j, const cmplx &value)
{
	if (i>=m_row || j>=m_column)
		return;
	V.replace(j + m_column*i, value);
}

uint Matrix::index(uint i, uint j) const
{
	if (i>=m_row || j>=m_column)
		return 0;
	return j + m_column*i;
}

Matrix::~Matrix()
{

}
