#include "../Grid/gridmanager.h"

GridManager::GridManager(int PastDomain, int FutureDomain, int TmpDomain, const Frame &F): GridManagerBase(PastDomain, FutureDomain, new Frame(F))
{
	m_Frame = GridManagerBase::m_Frame;
	if(TmpDomain>0)
	{
		m_tmpStackSize = TmpDomain;
	}
	else
	{
		m_tmpStackSize = 0;
	}

	for(int i=0; i<m_tmpStackSize; ++i)
	{
		m_tmpStack.append(new GridBase(m_Frame));
	}
}

void GridManager::swapStackTmp(const GridBase *S, const GridBase *T)
{

}

GridBase* GridManager::getTemporaryDomain(int i) const
{
	if(i<0 || i>=m_tmpStackSize)
	{
		return NULL;
	}
	return m_tmpStack.at(i);
}

GridManager::~GridManager()
{
	for(int i=0; i<m_tmpStackSize; ++i)
	{
		delete m_tmpStack.at(i);
	}
}
