#include "nonlinearaxis.h"

NonLinearAxis::NonLinearAxis(): public Axis
{
	Xmax = 0;
	Xmin = 0;
	nbPts = 1;
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

void NonLinearAxis::computeNumberPts()
{
	return (Xmax-Xmin)/(.1)+1
}
