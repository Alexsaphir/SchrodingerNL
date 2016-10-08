#ifndef COREMATRIX_H
#define COREMATRIX_H

#include <QVector>

#include "../type.h"

class CoreMatrix
{
public:
	CoreMatrix();
	CoreMatrix(uint n, bool isRowMatrix);
	CoreMatrix(uint row, uint column);
//Specific method for vector Matrix
	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);
//Generic method
	virtual cmplx getValue(uint i, uint j) const = 0;
	virtual void setValue(uint i, uint j, const cmplx &value) = 0;

	uint row() const;
	uint column() const;

	virtual ~CoreMatrix();
protected:
	uint m_row;
	uint m_column;
};

#endif // COREMATRIX_H
