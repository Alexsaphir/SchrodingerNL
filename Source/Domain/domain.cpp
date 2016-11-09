#include "domain.h"

Domain::Domain(): DomainBase(), m_Frame(NULL)
{
}

Domain::Domain(const Domain &D): DomainBase(new Frame(*D.m_Frame), D.m_BoundExt)
{
	m_Frame = GridBase::m_Frame;
}

Domain::Domain(const Axis *X, cmplx BoundExt): DomainBase(new Frame(X), BoundExt)
{
	m_Frame = GridBase::m_Frame;
}

Domain::Domain(const Axis *X, const Axis *Y, cmplx BoundExt): DomainBase(new Frame(X, Y), BoundExt)
{
	m_Frame = GridBase::m_Frame;
}

Domain::Domain(const Frame &F, cmplx BoundExt): DomainBase(new Frame(F), BoundExt)
{
	m_Frame = GridBase::m_Frame;
}

cmplx Domain::getBoundaryCondition(const Point &Pos) const
{
	return m_BoundExt;
}

Domain::~Domain()
{
	delete m_Frame;
}
