#include "include/axis.h"

Axis::Axis():Xmax(0), Xmin(0), Xstep(0), nbPts(0)
{

}

Axis::Axis(Type Xmn, Type Xmx, Type Xsp):Xmax(Xmx), Xmin(Xmn), Xstep(Xsp)
{
	nbPts = (Xmax-Xmin)/Xstep+1;
}

Type Axis::getAxisMax() const
{
	return Xmax;
}

Type Axis::getAxisMin() const
{
	return Xmin;
}

int Axis::getAxisN() const
{
	return nbPts;
}

Type Axis::getAxisStep() const
{
	return Xstep;
}
