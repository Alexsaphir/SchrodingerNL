#include "grid.h"

Grid::Grid(): GridBase()
{
	m_Frame = NULL;
}

Grid::Grid(const Frame &F): GridBase(new Frame(F))
{
	m_Frame = GridBase::m_Frame;
}

Grid::Grid(const Grid &G): GridBase(new Frame(*G.m_Frame))
{
	m_Frame = GridBase::m_Frame;
	for(int i=0; i<G.getSizeOfGrid(); ++i)
		this->setValue(i, G.getValue(i));
}

Grid::~Grid()
{
	delete m_Frame;
}
