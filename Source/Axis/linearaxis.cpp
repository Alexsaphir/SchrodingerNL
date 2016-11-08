#include "linearaxis.h"

LinearAxis::LinearAxis(): Axis(0, 0, 1)
{
	m_Xstep = 0;
}

LinearAxis::LinearAxis(Type Xmin, Type Xmax, Type Xstep): Axis(Xmin, Xmax, static_cast<int>((Xmax-Xmin)/Xstep)+1)
{
	m_Xstep = Xstep;
}

LinearAxis::LinearAxis(const LinearAxis &LA): Axis(LA)
{
	this->m_Xstep = LA.m_Xstep;
}

Type LinearAxis::getAxisStep() const
{
	return m_Xstep;
}

Type LinearAxis::getAxisStep(int nPt) const
{
	return m_Xstep;
}

Axis* LinearAxis::clone() const
{
	return new LinearAxis(*this);
}

LinearAxis::~LinearAxis()
{
}
