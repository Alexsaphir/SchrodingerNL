#include "nonlinearaxis.h"

NonLinearAxis::NonLinearAxis(): Axis(0, 0, 1)
{
	m_XStep.push_back(0);
}


NonLinearAxis::NonLinearAxis(Type Xmin, Type Xmax): Axis(Xmin, Xmax, computeNumberPts())
{
	for(int i=0; i<m_nbPts; ++i)
	{
		m_XStep.push_back(computeStep(i));
	}
}

NonLinearAxis::NonLinearAxis(const NonLinearAxis &NLA): Axis(NLA), m_XStep(NLA.m_XStep)
{
}

Type NonLinearAxis::getAxisStep(int nPt) const
{
	if (nPt>=m_nbPts)
		return 0.;
	return m_XStep.at(nPt);
}

Type NonLinearAxis::computeStep(int nPt) const
{
	if (nPt>=m_nbPts)
		return 0;
	return (m_Xmax-m_Xmin)/(static_cast<Type>(m_nbPts));
}

int NonLinearAxis::computeNumberPts() const
{
	return static_cast<int>((m_Xmax-m_Xmin)*10+1);
}

Axis* NonLinearAxis::clone() const
{
	return new NonLinearAxis(*this);
}

NonLinearAxis::~NonLinearAxis()
{

}
