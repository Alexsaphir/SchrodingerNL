#include "linearaxis.h"

LinearAxis::LinearAxis(): Axis(), Xmax(0), Xmin(0), Xstep(0), nbPts(0)
{
}

LinearAxis::LinearAxis(Type Xmn, Type Xmx, Type Xsp): Axis(), Xmax(Xmx), Xmin(Xmn), Xstep(Xsp)
{
	nbPts = (Xmax-Xmin)/Xstep+1;
}

LinearAxis::LinearAxis(const LinearAxis &LA)
{
	Xmax	= LA.Xmax;
	Xmin	= LA.Xmin;
	Xstep	= LA.Xstep;
	nbPts	= LA.nbPts;
}

Type LinearAxis::getAxisStep() const
{
	return Xstep;
}

Type LinearAxis::getAxisStep(uint nPt) const
{
	return Xstep;
}
