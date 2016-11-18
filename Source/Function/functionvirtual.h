#ifndef FUNCTIONVIRTUAL_H
#define FUNCTIONVIRTUAL_H

#include "../type.h"
#include "../Domain/Base/domainbase.h"
#include "../Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.h"
#include "../Matrix/Matrix/RowMatrix/rowmatrixvirtual.h"

class FunctionVirtual
{
public:
	FunctionVirtual();
	virtual ~FunctionVirtual();

	virtual FunctionVirtual* clone() const = 0;

	virtual cmplx computePartialDerivativeAt(CoreMatrix *C, int var) const =0;
	virtual cmplx computePartialDerivativeAt(DomainBase *D, int var) const =0;

	virtual cmplx evaluateAt(CoreMatrix *C) const = 0;
	virtual cmplx evaluateAt(DomainBase *D) const = 0;
	virtual cmplx evaluateAt(RowMatrixVirtual* R) const = 0;
	virtual cmplx evaluateAt(ColumnMatrixVirtual *C) const = 0;
};

#endif // FUNCTIONVIRTUAL_H
