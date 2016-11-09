#ifndef MATRIX_H
#define MATRIX_H

#include <QVector>

#include "../../type.h"
#include "../corematrix.h"

class Matrix: public CoreMatrix
{
public:
	Matrix(int row, int column);
	virtual ~Matrix();

	virtual cmplx getValue(int i, int j) const;
	virtual void setValue(int i, int j, const cmplx &value);

private:
	inline int index(int i, int j) const;

private:
	QVector<cmplx> V;
};

#endif // MATRIX_H

