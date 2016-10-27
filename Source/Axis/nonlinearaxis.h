#ifndef NONLINEARAXIS_H
#define NONLINEARAXIS_H

#include "axis.h"
#include "../type.h"

class NonLinearAxis: public Axis
{
public:
	NonLinearAxis();
	virtual Type getAxisStep(uint nPt) const;

private:
	virtual Type computeStep(uint nPt) const;
	virtual void computeNumberPts();
};

#endif // NONLINEARAXIS_H
