#ifndef NONLINEARAXIS_H
#define NONLINEARAXIS_H

#include <QVector>

#include "axis.h"
#include "../type.h"

class NonLinearAxis: public Axis
{
public:
	NonLinearAxis();
	NonLinearAxis(Type Xmin, Type Xmax);
	NonLinearAxis(const NonLinearAxis &NLA);
	~NonLinearAxis();

	Type getAxisStep(int nPt) const;

	Axis* clone() const;

protected:
	virtual Type computeStep(int nPt) const;
	virtual int computeNumberPts() const;

protected:
	QVector<Type> m_XStep;
};

#endif // NONLINEARAXIS_H
