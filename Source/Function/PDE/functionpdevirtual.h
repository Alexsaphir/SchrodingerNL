#ifndef FUNCTIONPDEVIRTUAL_H
#define FUNCTIONPDEVIRTUAL_H

#include "../functionvirtual.h"

#include "../../Grid/Base/gridmanagerbase.h"

class FunctionPDEVirtual: public FunctionVirtual
{
public:
	FunctionPDEVirtual(GridManagerBase *dmb);
	FunctionPDEVirtual(const FunctionPDEVirtual &F);
	~FunctionPDEVirtual();

	virtual FunctionVirtual* clone() const = 0;

	virtual cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const =0;
	virtual cmplx computePartialDerivativeAt(GridBase *D, int var) const =0;

	virtual cmplx evaluateAt(GridBase *D) const = 0;
	virtual cmplx evaluateAt(CoreMatrix *C) const = 0;

protected:
	GridManagerBase *m_gridManager;//this Pointer permit a direct access to the data but there is not any modification
	//It's just a Read access
};

#endif // FUNCTIONPDEVIRTUAL_H
