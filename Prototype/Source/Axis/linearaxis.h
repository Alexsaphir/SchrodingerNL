#ifndef LINEARAXIS_H
#define LINEARAXIS_H

#include "axis.h"
#include "../type.h"

class LinearAxis: public Axis
{
public:
	LinearAxis();
	LinearAxis(Type Xmin, Type Xmax, Type Xstep);
	LinearAxis(const LinearAxis &LA);
	~LinearAxis();

	Type getAxisStep() const;
	Type getAxisStep(int nPt) const;

	Axis* clone() const;

private:
	Type m_Xstep;
};

#endif // LINEARAXIS_H
