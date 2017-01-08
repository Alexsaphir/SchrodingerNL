#ifndef MATRIX_H
#define MATRIX_H

#include <QVector>

#include "../corematrix.h"

class Matrix: public CoreMatrix
{
public:
	Matrix(int row, int column);
	virtual ~Matrix();

	cmplx getValue(int i, int j) const;
	cmplx getValue(int i) const;
	void setValue(int i, int j, const cmplx &value);
	void setValue(int i, const cmplx &value);

private:
	inline int index(int i, int j) const;

private:
	QVector<cmplx> V;
};

#endif // MATRIX_H

