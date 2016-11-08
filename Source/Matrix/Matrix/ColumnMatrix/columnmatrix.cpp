#include "columnmatrix.h"

ColumnMatrix::ColumnMatrix(int row): ColumnMatrixVirtual(row)
{
	V.fill(cmplx(0,0), row);
}

cmplx ColumnMatrix::at(int i) const
{
	if(i>=m_row)
		return cmplx(0,0);
	else
		return V.at(i);
}

void ColumnMatrix::set(int i, const cmplx &value)
{
	if(i>=m_row)
		return;
	V.replace(i, value);
}

cmplx ColumnMatrix::getValue(int i, int j) const
{
	//Column Matrix => column=1 => j=1
	if(j!=0)
		return cmplx(0,0);
	else
		return at(i);
}

void ColumnMatrix::setValue(int i, int j, const cmplx &value)
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
