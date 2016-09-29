#ifndef AXIS_H
#define AXIS_H

#include "type.h"

class Axis
{
public:
	Axis();
	Axis(Type Xmn, Type Xmx, Type Xsp);

	Type getAxisMax() const;
	Type getAxisMin() const;
	Type getAxisStep() const;
	int getAxisN() const;
private:
	Type Xmax;
	Type Xmin;
	Type Xstep;
	int nbPts;
};

#endif // AXIS_H
