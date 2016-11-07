#ifndef NONLINEARAXIS_H
#define NONLINEARAXIS_H

#include "axis.h"
#include "../type.h"

class NonLinearAxis: public Axis
{
public:
	NonLinearAxis();
	NonLinearAxis(Type Xmn, Type Xmx);
	NonLinearAxis(const NonLinearAxis &NLA);

	virtual Type getAxisStep(uint nPt) const;

	virtual Axis* clone() const;
	~NonLinearAxis();

private:
	virtual Type computeStep(uint nPt) const;
	virtual uint computeNumberPts() const;
};

#endif // NONLINEARAXIS_H
