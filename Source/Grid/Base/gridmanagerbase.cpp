#include "gridmanagerbase.h"

GridManagerBase::GridManagerBase(int PastDomain, int FutureDomain, const Frame *F)
{
	if(PastDomain<0)
		PastDomain = 0;
	if (FutureDomain<0)
		FutureDomain = 0;

	m_offset = PastDomain;
	m_Size = PastDomain + FutureDomain + 1;

	for(int i=0; i<m_Size; ++i)
	{
		m_Stack.append(new GridBase(m_Frame));
	}
}

int GridManagerBase::getSizeStack() const
{
	return m_Size;
}

GridBase* GridManagerBase::getGrid(int i) const
{
	//Current == 0
	//Next == 1
	//Old == -1
	if( (m_offset+i)<0 || (m_offset+i)>=m_Size)
		return NULL;
	return m_Stack.at(m_offset+i);
}

GridBase* GridManagerBase::getCurrentGrid() const
{
	return getGrid(0);
}

GridBase* GridManagerBase::getNextGrid() const
{
	return getGrid(1);
}

GridBase* GridManagerBase::getOldGrid() const
{
	return getGrid(-1);
}

cmplx GridManagerBase::getValue(const Point &P, int t) const
{
	return getGrid(t)->getValue(P);
}

cmplx GridManagerBase::getValue(int i, int t) const
{
	return getGrid(t)->getValue(i);
}

void GridManagerBase::switchGrid()
{
	if(m_Size == 0)
		return;
	m_Stack.push_back(m_Stack.first());
	m_Stack.removeFirst();
}

ColumnDataProxy* GridManagerBase::getColumnAtTime(int t) const
{
	return getGrid(t)->getColumn();
}

RowDataProxy* GridManagerBase::getRowAtTime(int t) const
{
	return getGrid(t)->getRow();
}

GridManagerBase::~GridManagerBase()
{
	for(int i=0; i<m_Size; ++i)
	{
		delete m_Stack.at(i);
	}
}
