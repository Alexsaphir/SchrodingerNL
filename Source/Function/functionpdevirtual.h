#ifndef FUNCTIONPDEVIRTUAL_H
#define FUNCTIONPDEVIRTUAL_H

#include "functionvirtual.h"

#include "../Domain/Base/domainmanagerbase.h"

class FunctionPDEVirtual
{
public:
	FunctionPDEVirtual();
	FunctionPDEVirtual(DomainManagerBase *dmb);
	FunctionPDEVirtual(const FunctionPDEVirtual &F);
	~FunctionPDEVirtual();

	virtual FunctionVirtual* clone() const = 0;

	virtual cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const =0;
	virtual cmplx computePartialDerivativeAt(DomainBase *D, int var) const =0;

	virtual cmplx evaluateAt(DomainBase *D) const = 0;
	virtual cmplx evaluateAt(CoreMatrix *C) const = 0;
	virtual cmplx evaluateAt(RowMatrixVirtual* R) const = 0;
	virtual cmplx evaluateAt(ColumnMatrixVirtual *C) const = 0;

protected:
	DomainManagerBase *m_domainManager;//this Pointer permit a direct access to the data but there is not any modification
	//It's just a Read access
};

#endif // FUNCTIONPDEVIRTUAL_H
