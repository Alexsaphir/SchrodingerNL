#include "corematrix.h"

CoreMatrix::CoreMatrix(): m_row(0), m_column(0)
{
}

CoreMatrix::CoreMatrix(int n, bool isRowMatrix)
{
	if(isRowMatrix)
	{
		m_row	 = 1;
		m_column = n;
	}
	else
	{
		m_row	 = n;
		m_column = 1;
	}
}

CoreMatrix::CoreMatrix(int row, int column): m_row(row), m_column(column)
{
}

int CoreMatrix::column() const
{
	return m_column;
}

int CoreMatrix::row() const
{
	return m_row;
}

CoreMatrix::~CoreMatrix()
{
}

cmplx CoreMatrix::at(int i) const
{
	return cmplx(0,0);//If method if not redefined this isn't a vector
}

void CoreMatrix::set(int i, const cmplx &value)
{
	return;
}
