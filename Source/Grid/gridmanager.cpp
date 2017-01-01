#include "../Grid/gridmanager.h"

GridManager::GridManager(int PastDomain, int FutureDomain, const Frame &F): GridManagerBase(PastDomain, FutureDomain, new Frame(F))
{
	m_Frame = GridManagerBase::m_Frame;
}


GridManager::~GridManager()
{
}
