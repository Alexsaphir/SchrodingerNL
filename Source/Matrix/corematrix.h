#ifndef COREMATRIX_H
#define COREMATRIX_H

#include <QVector>

#include "../type.h"

class CoreMatrix
{
public:
	CoreMatrix();
	CoreMatrix(int n, bool isRowMatrix);
	CoreMatrix(int row, int column);
	virtual ~CoreMatrix();

//Specific method for vector Matrix
	virtual cmplx at(int i) const;
	virtual void set(int i, const cmplx &value);
//Generic method
	virtual cmplx getValue(int i, int j) const = 0;
	virtual void setValue(int i, int j, const cmplx &value) = 0;

	virtual int row() const;
	virtual int column() const;
protected:
	int m_row;
	int m_column;
};

#endif // COREMATRIX_H
