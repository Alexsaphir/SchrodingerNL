#ifndef LINEARAXIS_H
#define LINEARAXIS_H

#include "axis.h"
#include "../type.h"

class LinearAxis: public Axis
{
public:
	LinearAxis();
	LinearAxis(Type Xmn, Type Xmx, Type Xsp);
	LinearAxis(const LinearAxis &LA);

	virtual Type getAxisStep() const;
	virtual Type getAxisStep(uint nPt) const;

private:
	Type Xstep;
};

#endif // LINEARAXIS_H
