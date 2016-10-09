#include "columnmatrix.h"

ColumnMatrix::ColumnMatrix(uint row): ColumnMatrixVirtual(row)
{
	V.fill(cmplx(0,0), row);
}

cmplx ColumnMatrix::at(uint i) const
{
	if(i>=m_row)
		return (0,0);
	else
		return V.at(i);
}

void ColumnMatrix::set(uint i, const cmplx &value)
{
	if(i>=m_row)
		return;
	V.replace(i, value);
}

cmplx ColumnMatrix::getValue(uint i, uint j) const
{
	//Column Matrix => column=1 => j=1
	if(j!=1)
		return cmplx(0,0);
	else
		return at(i);
}

void ColumnMatrix::setValue(uint i, uint j, const cmplx &value)
{
	if(j!=1)
		return;
	if(i>=m_row)
		return;
	V.replace(i, value);
}


ColumnMatrix::~ColumnMatrix()
{

}
