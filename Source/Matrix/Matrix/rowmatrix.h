#ifndef ROWMATRIX_H
#define ROWMATRIX_H

#include <QVector>

#include "../corematrix.h"
#include "../../type.h"

class RowMatrix: public CoreMatrix
{
public:
	RowMatrix(uint column);

	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);

	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);

	virtual ~RowMatrix();
private:
	QVector<cmplx> V;
};

#endif // ROWMATRIX_H
