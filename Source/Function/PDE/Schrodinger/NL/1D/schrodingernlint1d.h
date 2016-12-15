#ifndef SCHRODINGERNLINT1D_H
#define SCHRODINGERNLINT1D_H

#include "../../../functionpdevirtual.h"

class SchrodingerNLInt1D: public FunctionPDEVirtual
{
public:
	SchrodingerNLInt1D(int i, GridManagerBase *G, Type dx, Type dt, Type k);
	~SchrodingerNLInt1D();
	FunctionVirtual* clone() const;

	cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const;
	cmplx computePartialDerivativeAt(GridBase *D, int var) const;

	cmplx evaluateAt(GridBase *D) const;
	cmplx evaluateAt(CoreMatrix *C) const;

protected:
	cmplx f(CoreMatrix *C) const;
	cmplx df(CoreMatrix *C, int var) const;
	int m_intIndex;
	cmplx m_idtk;
	cmplx m_idtdiv2dx;
};

#endif // SCHRODINGERNLINT1D_H
