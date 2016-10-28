#include "nonlinearaxis.h"

NonLinearAxis::NonLinearAxis(): Axis(), Xmax(0), Xmin(0), nbPts(1)
{
}

NonLinearAxis::NonLinearAxis(Type Xmn, Type Xmx): Xmax(Xmx), Xmin(Xmn)
{
	nbPts = computeNumberPts();
}

NonLinearAxis::NonLinearAxis(const NonLinearAxis &NLA)
{
	Xmax = NLA.Xmax;
	Xmin = NLA.Xmin;
	nbPts = NLA.nbPts;
}

Type NonLinearAxis::getAxisStep(uint nPt) const
{
	if (nPt>=nbPts)
		return 0.;
	return computeStep(nPt);
}

Type NonLinearAxis::computeStep(uint nPt) const
{
	return (Xmax-Xmin)/((Type)nbPts);
}

uint NonLinearAxis::computeNumberPts() const
{
	return (Xmax-Xmin)/(.1)+1;
}

Axis NonLinearAxis::clone() const
{
	return new NonLinearAxis(*this);
}
