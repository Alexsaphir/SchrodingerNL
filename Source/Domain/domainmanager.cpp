#include "domainmanager.h"

DomainManager::DomainManager(int PastDomain, int FutureDomain, const Frame &F, cmplx BoundExt): DomainManagerBase(PastDomain, FutureDomain, new Frame(F), BoundExt)
{
	m_Frame = DomainManagerBase::m_Frame;

	CurrProxy = new ColumnDataProxy(getCurrentDomain());
	NextProxy = new ColumnDataProxy(getNextDomain());
}

ColumnDataProxy* DomainManager::getCurrentColumn() const
{
	return CurrProxy;
}

ColumnDataProxy* DomainManager::getNextColumn() const
{
	return NextProxy;
}

DomainManager::~DomainManager()
{
	delete m_Frame;
}
