#ifndef AXIS_H
#define AXIS_H

#include "../type.h"


class Axis
{
public:
	Axis();

	Type getAxisMax() const;
	Type getAxisMin() const;
	virtual Type getAxisStep() const;
	virtual Type getAxisStep(uint nPt) const =0;
	uint getAxisN() const;
protected:
	Type Xmax;
	Type Xmin;
	uint nbPts;
};

#endif // AXIS_H
