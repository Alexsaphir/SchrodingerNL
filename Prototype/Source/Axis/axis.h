#ifndef AXIS_H
#define AXIS_H

#include "../type.h"

class Axis
{
public:
	Axis();
	Axis(Type Xmin, Type Xmax, int nbPts);
	Axis(const Axis &A);
	virtual ~Axis();

	Type getAxisMax() const;
	Type getAxisMin() const;
	int getAxisN() const;

	virtual Type getAxisStep() const;
	virtual Type getAxisStep(int nPt) const =0;

	virtual Axis* clone() const = 0;
protected:
	Type m_Xmin;
	Type m_Xmax;
	int m_nbPts;
};

#endif // AXIS_H
