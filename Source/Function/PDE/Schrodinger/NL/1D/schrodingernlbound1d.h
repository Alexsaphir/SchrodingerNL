#ifndef SCHRODINGERNLBOUND1D_H
#define SCHRODINGERNLBOUND1D_H

#include "../../../functionpdevirtual.h"

class SchrodingerNLBound1D: public FunctionPDEVirtual
{
public:
	SchrodingerNLBound1D(int i, GridManagerBase *G);
	SchrodingerNLBound1D(const SchrodingerNLBound1D &E);

	FunctionVirtual* clone() const;

	cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const;
	cmplx computePartialDerivativeAt(GridBase *D, int var) const;

	cmplx evaluateAt(GridBase *D) const;
	cmplx evaluateAt(CoreMatrix *C) const;

protected:
	cmplx f(CoreMatrix *C) const;
	cmplx df(CoreMatrix *C, int var) const;
	int m_boundIndex;
};

#endif // SCHRODINGERNLBOUND1D_H
