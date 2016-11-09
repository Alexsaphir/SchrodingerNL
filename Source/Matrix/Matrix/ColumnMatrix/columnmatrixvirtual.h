#ifndef COLUMNMATRIXVIRTUAL_H
#define COLUMNMATRIXVIRTUAL_H

#include "../../corematrix.h"
#include "../../../type.h"

class ColumnMatrixVirtual: public CoreMatrix
{
public:
	ColumnMatrixVirtual();
	ColumnMatrixVirtual(int row);
	virtual ~ColumnMatrixVirtual();

	virtual cmplx at(int i) const = 0;
	virtual void set(int i, const cmplx &value) = 0;

	virtual cmplx getValue(int i, int j) const = 0;
	virtual void setValue(int i, int j, const cmplx &value) = 0;
};

#endif // COLUMNMATRIXVIRTUAL_H
