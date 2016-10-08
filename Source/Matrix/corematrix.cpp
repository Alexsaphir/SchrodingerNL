#include "corematrix.h"

CoreMatrix::CoreMatrix(): m_row(0), m_column(0)
{
}

CoreMatrix::CoreMatrix(uint row): m_row(row), m_column(1)
{

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
