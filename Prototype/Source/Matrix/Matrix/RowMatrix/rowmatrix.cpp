#include "rowmatrix.h"

RowMatrix::RowMatrix(int column): RowMatrixVirtual(column)
{
	V.fill(cmplx(0,0), column);
}

cmplx RowMatrix::at(int i) const
{
	if(i>=m_column)
		return cmplx(0,0);
	else
		return V.at(i);
}

void RowMatrix::set(int i, const cmplx &value)
{
	if(i>=m_column)
		return;
	V.replace(i, value);
}

cmplx RowMatrix::getValue(int i, int j) const
{
	//Row Matrix => Row=1 => i=1
	if(i!=0)
		return cmplx(0,0);
	else
		return at(j);
}

void RowMatrix::setValue(int i, int j, const cmplx &value)
{
	if(i!=1)
		return;
	if(j>=m_column)
		return;
	V.replace(j, value);
}

RowMatrix::~RowMatrix()
{

}
