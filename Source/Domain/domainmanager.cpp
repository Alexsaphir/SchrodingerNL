#include "domainmanager.h"

DomainManager::DomainManager(int PastDomain, int FutureDomain, const Frame &F, cmplx BoundExt): DomainManagerBase(PastDomain, FutureDomain, new Frame(F), BoundExt)
{
	m_Frame = DomainManagerBase::m_Frame;
}

DomainManager::~DomainManager()
{
	delete m_Frame;
}
