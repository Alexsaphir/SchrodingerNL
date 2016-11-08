#include "domain.h"

Domain::Domain(): DomainPrivate(), m_Frame(NULL)
{
}

Domain::Domain(const Domain &D): DomainPrivate(new Frame(*D.m_Frame), D.m_BoundExt)
{
	m_Frame = getFrame();
}

Domain::Domain(const Axis *X, cmplx BoundExt): DomainPrivate(new Frame(X), BoundExt)
{
	m_Frame = getFrame();
}

Domain::Domain(const Axis *X, const Axis *Y, cmplx BoundExt): DomainPrivate(new Frame(X, Y), BoundExt)
{
	m_Frame = getFrame();
}

Domain::Domain(const Frame &F, cmplx BoundExt): DomainPrivate(new Frame(F), BoundExt)
{
	m_Frame = getFrame();
}

cmplx Domain::getBoundaryCondition(const Point &Pos) const
{
	return m_BoundExt;
}

Domain::~Domain()
{
	delete m_Frame;
}
