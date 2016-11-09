#include "rowdataproxy.h"

RowDataProxy::RowDataProxy(): RowMatrixVirtual()
{
	m_domain = NULL;
}

RowDataProxy::RowDataProxy(DomainBase *D): RowMatrixVirtual(), m_domain(D)
{

}

cmplx RowDataProxy::at(int i) const
{
	if(i>=column())
		return cmplx(0,0);
	return m_domain->getValue(i);
}

int RowDataProxy::column() const
{
	if(!m_domain)
		return 0;//m_domain == Null
	return m_domain->getSizeOfGrid();
}

cmplx RowDataProxy::getValue(int i, int j) const
{
	if(i!=0)
		return cmplx(0,0);
	return at(j);
}

int RowDataProxy::row() const
{
	if(column()==0)
		return 0;
	return 1;
}

void RowDataProxy::set(int i, const cmplx &value)
{
	if(i>=column())
		return;
	m_domain->setValue(i, value);
}

void RowDataProxy::setDomain(DomainBase *D)
{
	m_domain = D;
}

void RowDataProxy::setValue(int i, int j, const cmplx &value)
{
	if(i!=1)
		return;
	set(j, value);
}

RowDataProxy::~RowDataProxy()
{

}
