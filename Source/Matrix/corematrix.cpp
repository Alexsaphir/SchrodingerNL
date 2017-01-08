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

cmplx CoreMatrix::getValue(int i) const
{
	return at(i);
}

void CoreMatrix::setValue(int i, const cmplx &value)
{
	set(i, value);
}

cmplx CoreMatrix::at(int i) const
{
	return cmplx(0,0);//If method if not redefined this isn't a vector
}

void CoreMatrix::set(int i, const cmplx &value)
{
	return;
}

void CoreMatrix::operator +=(CoreMatrix const& M)
{
	if (this->m_column != M.m_column)
		return;
	if (this->m_row != M.m_row)
		return;
	for(int i=0; i<(this->m_row*this->m_column); ++i)
	{
		this->setValue(i, this->getValue(i)+M.getValue(i));
	}
}

void CoreMatrix::operator -=(CoreMatrix const& M)
{
	if (this->m_column != M.m_column)
		return;
	if (this->m_row != M.m_row)
		return;
	for(int i=0; i<(this->m_row*this->m_column); ++i)
	{
		this->setValue(i, this->getValue(i)-M.getValue(i));
	}
}

CoreMatrix::~CoreMatrix()
{
}
