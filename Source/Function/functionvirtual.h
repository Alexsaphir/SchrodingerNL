#ifndef FUNCTIONVIRTUAL_H
#define FUNCTIONVIRTUAL_H

#include "../type.h"
#include "../Grid/Base/gridbase.h"

class FunctionVirtual
{
public:
	FunctionVirtual();
	virtual ~FunctionVirtual();

	virtual FunctionVirtual* clone() const = 0;

	virtual cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const =0;
	virtual cmplx computePartialDerivativeAt(GridBase *D, int var) const =0;

	virtual cmplx evaluateAt(CoreMatrix *C) const = 0;
	virtual cmplx evaluateAt(GridBase *D) const = 0;
};

#endif // FUNCTIONVIRTUAL_H
