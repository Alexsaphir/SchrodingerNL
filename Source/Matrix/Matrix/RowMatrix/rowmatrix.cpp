#include "rowmatrix.h"

RowMatrix::RowMatrix(uint column): RowMatrixVirtual(column)
{
	V.fill(cmplx(0,0), column);
}

cmplx RowMatrix::at(uint i) const
{
	if(i>=m_column)
		return cmplx(0,0);
	else
		return V.at(i);
}

void RowMatrix::set(uint i, const cmplx &value)
{
	if(i>=m_column)
		return;
	V.replace(i, value);
}

cmplx RowMatrix::getValue(uint i, uint j) const
{
	//Row Matrix => Row=1 => i=1
	if(i!=1)
		return cmplx(0,0);
	else
		return at(j);
}

void RowMatrix::setValue(uint i, uint j, const cmplx &value)
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
