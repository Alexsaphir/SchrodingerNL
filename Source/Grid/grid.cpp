#include "grid.h"

Grid::Grid(): GridPrivate()
{
	m_Frame = NULL;
}

Grid::Grid(const Frame &F): GridPrivate(new Frame(F))
{
	m_Frame = getFrame();
}

Grid::Grid(const Grid &G): GridPrivate(new Frame(*G.m_Frame))
{
	m_Frame = getFrame();
	for(int i=0; i<G.getSizeOfGrid(); ++i)
		this->setValue(i, G.getValue(i));
}

Grid::~Grid()
{
	delete m_Frame;
}
