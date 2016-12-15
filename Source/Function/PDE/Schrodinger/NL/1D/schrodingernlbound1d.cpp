#include "schrodingernlbound1d.h"

SchrodingerNLBound1D::SchrodingerNLBound1D(int i, GridManagerBase *G): FunctionPDEVirtual(G), m_boundIndex(i)
{
}

SchrodingerNLBound1D::SchrodingerNLBound1D(const SchrodingerNLBound1D &E): FunctionPDEVirtual(E.m_gridManager), m_boundIndex(E.m_boundIndex)
{
}

FunctionVirtual* SchrodingerNLBound1D::clone() const
{
	return new SchrodingerNLBound1D(*this);
}

cmplx SchrodingerNLBound1D::computePartialDerivativeAt(CoreMatrix *C, int var) const
{
	if(var != m_boundIndex)
		return 0;
	else return 1;
}

cmplx SchrodingerNLBound1D::computePartialDerivativeAt(GridBase *D, int var) const
{
	return computePartialDerivativeAt(D->getColumn(), var);
}


cmplx SchrodingerNLBound1D::evaluateAt(CoreMatrix *C) const
{
	return f(C);
}

cmplx SchrodingerNLBound1D::evaluateAt(GridBase *D) const
{
	return evaluateAt(D->getColumn());
}

cmplx SchrodingerNLBound1D::f(CoreMatrix *C) const
{
	return C->at(m_boundIndex);
}

cmplx SchrodingerNLBound1D::df(CoreMatrix *C, int var) const
{
	if(var != m_boundIndex)
		return cmplx(0,0);
	return cmplx(1.,0);
}
