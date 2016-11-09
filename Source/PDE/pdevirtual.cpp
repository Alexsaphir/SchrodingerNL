#include "pdevirtual.h"

PDEVirtual::PDEVirtual(): m_Frame(NULL), m_Space(NULL)
{
}

PDEVirtual::PDEVirtual(const Frame &F): m_Frame(new Frame(F))
{
	m_Space = new DomainManagerBase(0, 0, m_Frame, 0.);
}

PDEVirtual::PDEVirtual(const Frame &F, int Past, int Future, cmplx BoundExt)
{
	m_Frame = new Frame(F);
	m_Space = new DomainManagerBase(Past, Future, m_Frame, BoundExt);
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
