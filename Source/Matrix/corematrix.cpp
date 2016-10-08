#include "corematrix.h"

CoreMatrix::CoreMatrix(): m_row(0), m_column(0)
{
}

CoreMatrix::CoreMatrix(uint row, uint column): m_row(row), m_column(column)
{
}

//cmplx CoreMatrix::getValue(uint i, uint j) const
//{
//	return cmplx(0,0);
//}

void CoreMatrix::setValue(uint i, uint j, const cmplx &value)
{
	//Do nothing
}


uint CoreMatrix::column() const
{
	return m_column;
}

uint CoreMatrix::row() const
{
	return m_row;
}

CoreMatrix::~CoreMatrix()
{

}
