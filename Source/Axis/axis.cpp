#include "axis.h"

Axis::Axis()
{
}

Type Axis::getAxisMax() const
{
	return Xmax;
}

Type Axis::getAxisMin() const
{
	return Xmin;
}

uint Axis::getAxisN() const
{
	return nbPts;
}

Type Axis::getAxisStep() const
{
	return 0.;
}
