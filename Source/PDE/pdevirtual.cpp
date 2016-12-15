#include "pdevirtual.h"

PDEVirtual::PDEVirtual(): m_Frame(NULL), m_Space(NULL)
{
}

PDEVirtual::PDEVirtual(const Frame &F): m_Frame(new Frame(F))
{
	m_Space = new GridManagerBase(0, 0, m_Frame);
}

PDEVirtual::PDEVirtual(const Frame &F, int Past, int Future)
{
	m_Frame = new Frame(F);
	m_Space = new GridManagerBase(Past, Future, m_Frame);
}

cmplx PDEVirtual::at(const Point &P) const
{
	return cmplx(0,0);
}

PDEVirtual::~PDEVirtual()
{
	delete m_Frame;
	delete m_Space;
}
