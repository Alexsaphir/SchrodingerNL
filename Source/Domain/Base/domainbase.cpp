#include "domainbase.h"

DomainBase::DomainBase(): GridBase(), m_BoundExt(cmplx(0,0)), m_ProxyColumn(NULL), m_ProxyRow(NULL)
{

}

DomainBase::DomainBase(const Frame *F, cmplx BoundExt): GridBase(F), m_BoundExt(BoundExt)
{
	m_ProxyColumn = new ColumnDataProxy(this);
	m_ProxyRow = new RowDataProxy(this);
}

cmplx DomainBase::getValue(const Point &Pos) const
{
	int i = getIndexFromPos(Pos);
	if(i == -1)
		return cmplx(0,0);
	return  GridBase::getValue(i);
}

cmplx DomainBase::getValue(int i) const
{
	if((i<0) || (i>=getSizeOfGrid()))
		return m_BoundExt;
	return  GridBase::getValue(i);
}

ColumnDataProxy* DomainBase::getColumn() const
{
	return m_ProxyColumn;
}

RowDataProxy* DomainBase::getRow() const
{
	return m_ProxyRow;
}

DomainBase::~DomainBase()
{
	delete m_ProxyColumn;
	delete m_ProxyRow;
}
