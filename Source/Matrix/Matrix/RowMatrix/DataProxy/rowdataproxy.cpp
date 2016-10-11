#include "rowdataproxy.h"

RowDataProxy::RowDataProxy(): RowMatrixVirtual()
{
	m_domain = NULL;
}

RowDataProxy::RowDataProxy(Domain *D): RowMatrixVirtual(), m_domain(D)
{

}

cmplx RowDataProxy::at(uint i) const
{
	if(i>=column())
		return cmplx(0,0);
	return m_domain->getValue(i);
}

uint RowDataProxy::column() const
{
	if(!m_domain)
		return 0;//m_domain == Null
	return m_domain->getN();
}

cmplx RowDataProxy::getValue(uint i, uint j) const
{
	if(i!=0)
		return cmplx(0,0);
	return at(j);
}

uint RowDataProxy::row() const
{
	if(column()==0)
		return 0;
	return 1;
}

void RowDataProxy::set(uint i, const cmplx &value)
{
	if(i>=column())
		return;
	m_domain->setValue(i, value);
}

void RowDataProxy::setDomain(Domain *D)
{
	m_domain = D;
}

void RowDataProxy::setValue(uint i, uint j, const cmplx &value)
{
	if(i!=1)
		return;
	set(j, value);
}

RowDataProxy::~RowDataProxy()
{

}
