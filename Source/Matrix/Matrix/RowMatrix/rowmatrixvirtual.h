#ifndef ROWMATRIXVIRTUAL_H
#define ROWMATRIXVIRTUAL_H

#include "../../corematrix.h"
#include "../../../type.h"

class RowMatrixVirtual: public CoreMatrix
{
public:
	RowMatrixVirtual();


	virtual cmplx at(uint i) const = 0;
	virtual void set(uint i, const cmplx &value) = 0;

	virtual cmplx getValue(uint i, uint j) const = 0;
	virtual void setValue(uint i, uint j, const cmplx &value) = 0;

	~RowMatrixVirtual();
};

#endif // ROWMATRIXVIRTUAL_H
