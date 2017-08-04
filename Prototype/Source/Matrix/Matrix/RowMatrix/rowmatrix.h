#ifndef ROWMATRIX_H
#define ROWMATRIX_H

#include <QVector>

#include "rowmatrixvirtual.h"

class RowMatrix: public RowMatrixVirtual
{
public:
	RowMatrix(int column);
	virtual ~RowMatrix();

	virtual cmplx at(int i) const;
	virtual void set(int i, const cmplx &value);

	virtual cmplx getValue(int i, int j) const;
	virtual void setValue(int i, int j, const cmplx &value);

private:
	QVector<cmplx> V;
};

#endif // ROWMATRIX_H
