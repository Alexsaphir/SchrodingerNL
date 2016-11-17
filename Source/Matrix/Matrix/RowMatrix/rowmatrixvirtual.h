#ifndef ROWMATRIXVIRTUAL_H
#define ROWMATRIXVIRTUAL_H

#include "../../corematrix.h"

class RowMatrixVirtual: public CoreMatrix
{
public:
	RowMatrixVirtual();
	RowMatrixVirtual(int column);
	virtual ~RowMatrixVirtual();

	virtual cmplx at(int i) const = 0;
	virtual void set(int i, const cmplx &value) = 0;

	virtual cmplx getValue(int i, int j) const = 0;
	virtual void setValue(int i, int j, const cmplx &value) = 0;
};

#endif // ROWMATRIXVIRTUAL_H
