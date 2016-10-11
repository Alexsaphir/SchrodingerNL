#include "columndataproxy.h"

ColumnDataProxy::ColumnDataProxy(): ColumnMatrixVirtual()
{
	m_domain = NULL;
}

ColumnDataProxy::ColumnDataProxy(Domain *D): ColumnMatrixVirtual(), m_domain(D)
{

}

cmplx ColumnDataProxy::at(uint i) const
{
	if(i>=row())
		return cmplx(0,0);
	return m_domain->getValue(i);
}

uint ColumnDataProxy::column() const
{
	if(row()==0)
		return 0;
	else
		return 1;
}

cmplx ColumnDataProxy::getValue(uint i, uint j) const
{
	if(j!=0)
		return cmplx(0,0);
	return at(i);
}

uint ColumnDataProxy::row() const
{

	if(!m_domain)
		return 0;//m_domain are set to NULL
	return m_domain->getN();
}

void ColumnDataProxy::set(uint i, const cmplx &value)
{
	if(i>=row())
		return;
	m_domain->setValue(i, value);
}

void ColumnDataProxy::setDomain(Domain *D)
{
	m_domain = D;
}

void ColumnDataProxy::setValue(uint i, uint j, const cmplx &value)
{
	qDebug() << "call";
	if(j!=1)
		return;
	set(i, value);
}

ColumnDataProxy::~ColumnDataProxy()
{

}
