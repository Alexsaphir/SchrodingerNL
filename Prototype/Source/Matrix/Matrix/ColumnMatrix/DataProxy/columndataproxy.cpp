#include "columndataproxy.h"

ColumnDataProxy::ColumnDataProxy(): ColumnMatrixVirtual(0)
{
	m_domain = NULL;
}

ColumnDataProxy::ColumnDataProxy(GridBase *D): ColumnMatrixVirtual(), m_domain(D)
{
	m_column = 1;
	m_row = D->getSizeOfGrid();
}

cmplx ColumnDataProxy::at(int i) const
{
	if(i>=row())
		return cmplx(0,0);
	return m_domain->getValue(i);
}

int ColumnDataProxy::column() const
{
	if(row()==0)
		return 0;
	else
		return 1;
}

cmplx ColumnDataProxy::getValue(int i, int j) const
{
	if(j!=0)
		return cmplx(0,0);
	return at(i);
}

int ColumnDataProxy::row() const
{

	if(!m_domain)
		return 0;//m_domain are set to NULL
	return m_domain->getSizeOfGrid();
}

void ColumnDataProxy::set(int i, const cmplx &value)
{
	if(i>=row())
		return;
	m_domain->setValue(i, value);
}

void ColumnDataProxy::setValue(int i, int j, const cmplx &value)
{
	if(j!=1)
		return;
	set(i, value);
}

ColumnDataProxy::~ColumnDataProxy()
{
}
