#ifndef COLUMNMATRIX_H
#define COLUMNMATRIX_H

#include <QVector>

#include "../corematrix.h"
#include "../../type.h"

class ColumnMatrix: public CoreMatrix
{
public:
	ColumnMatrix(uint row);

	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);

	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);


	virtual ~ColumnMatrix();

private:
	QVector<cmplx> V;
};

#endif // COLUMNMATRIX_H
