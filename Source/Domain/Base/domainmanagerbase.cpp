#include "domainmanagerbase.h"

DomainManagerBase::DomainManagerBase(): m_Frame(NULL), m_Size(0), m_offset(0), m_CurrProxy(NULL), m_NextProxy(NULL)
{
}

DomainManagerBase::DomainManagerBase(int PastDomain, int FutureDomain, const Frame *F, cmplx BoundExt): m_Frame(F)
{
	if(PastDomain<0)
		PastDomain = 0;
	if (FutureDomain<0)
		FutureDomain = 0;

	m_offset = PastDomain;
	m_Size = PastDomain + FutureDomain + 1;

	for(int i=0; i<m_Size; ++i)
	{
		m_Stack.append(new DomainBase(m_Frame, BoundExt));
		m_ProxyColumn.append(new ColumnDataProxy(m_Stack.last()));
		m_ProxyRow.append(new RowDataProxy(m_Stack.last()));
	}

	m_CurrProxy = new ColumnDataProxy();
	m_NextProxy = new ColumnDataProxy();
}

int DomainManagerBase::getSizeStack()
{
	return m_Size;
}

DomainBase* DomainManagerBase::getDomain(int i) const
{
	//Current == 0
	//Next == 1
	//Old == -1
	if( (m_offset+i)<0 || (m_offset+i)>=m_Size)
		return NULL;
	return m_Stack.at(m_offset+i);
}

DomainBase* DomainManagerBase::getCurrentDomain() const
{
	return getDomain(0);
}

DomainBase* DomainManagerBase::getNextDomain() const
{
	return getDomain(1);
}

DomainBase* DomainManagerBase::getOldDomain() const
{
	return getDomain(-1);
}

cmplx DomainManagerBase::getValue(const Point &P, int t) const
{
	return getDomain(t)->getValue(P);
}

void DomainManagerBase::switchDomain()
{
	if(m_Size == 0)
		return;
	m_Stack.push_back(m_Stack.first());
	m_ProxyColumn.push_back(m_ProxyColumn.first());
	m_ProxyRow.push_back(m_ProxyRow.first());
	m_Stack.removeFirst();
	m_ProxyColumn.removeFirst();
	m_ProxyRow.removeFirst();
}

ColumnDataProxy* DomainManagerBase::getCurrentColumn() const
{
	return getColumnAtTime(0);
}

ColumnDataProxy* DomainManagerBase::getNextColumn() const
{
	return getColumnAtTime(1);
}

ColumnDataProxy* DomainManagerBase::getColumnAtTime(int t) const
{
	//Current == 0
	//Next == 1
	//Old == -1
	if( (m_offset+i)<0 || (m_offset+i)>=m_Size)
		return NULL;
	return m_ProxyColumn.at(m_offset+i);
}

RowDataProxy* DomainManagerBase::getRowAtTime(int t) const
{
	//Current == 0
	//Next == 1
	//Old == -1
	if( (m_offset+i)<0 || (m_offset+i)>=m_Size)
		return NULL;
	return m_ProxyRow.at(m_offset+i);
}

DomainManagerBase::~DomainManagerBase()
{
	delete m_CurrProxy;
	delete m_NextProxy;
	for(int i=0; i<m_Size; ++i)
	{
		delete m_Stack.at(i);
		delete m_ProxyColumn.at(i);
		delete m_ProxyRow.at(i);
	}
}
