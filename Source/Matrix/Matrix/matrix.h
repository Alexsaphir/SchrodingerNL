#ifndef MATRIX_H
#define MATRIX_H

#include <QVector>

#include "../../type.h"
#include "../corematrix.h"

class Matrix: public CoreMatrix
{
public:
	Matrix(uint row, uint column);

	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);

	virtual ~Matrix();
private:
	inline uint index(uint i, uint j) const;
private:
	QVector<cmplx> V;
};

#endif // MATRIX_H

