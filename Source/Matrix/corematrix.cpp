#include "corematrix.h"

CoreMatrix::CoreMatrix(): m_row(0), m_column(0)
{
}

CoreMatrix::CoreMatrix(uint n, bool isRowMatrix)
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

CoreMatrix::CoreMatrix(uint row, uint column): m_row(row), m_column(column)
{
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
