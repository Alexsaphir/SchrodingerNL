#ifndef COLUMNMATRIX_H
#define COLUMNMATRIX_H

#include <QVector>

#include "columnmatrixvirtual.h"
#include "../../../type.h"

class ColumnMatrix: public ColumnMatrixVirtual
{
public:
	ColumnMatrix(int row);
	virtual ~ColumnMatrix();

	virtual cmplx at(int i) const;
	virtual void set(int i, const cmplx &value);

	virtual cmplx getValue(int i, int j) const;
	virtual void setValue(int i, int j, const cmplx &value);

private:
	QVector<cmplx> V;
};

#endif // COLUMNMATRIX_H
