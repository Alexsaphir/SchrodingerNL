#ifndef COLUMNMATRIXVIRTUAL_H
#define COLUMNMATRIXVIRTUAL_H

#include "../../corematrix.h"
#include "../../../type.h"

class ColumnMatrixVirtual: public CoreMatrix
{
public:
	ColumnMatrixVirtual();
	ColumnMatrixVirtual(uint row);

	virtual cmplx at(uint i) const = 0;
	virtual void set(uint i, const cmplx &value) = 0;

	virtual cmplx getValue(uint i, uint j) const = 0;
	virtual void setValue(uint i, uint j, const cmplx &value) = 0;

	virtual ~ColumnMatrixVirtual();
};

#endif // COLUMNMATRIXVIRTUAL_H
