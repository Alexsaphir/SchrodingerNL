#ifndef COREMATRIX_H
#define COREMATRIX_H

#include <QVector>

#include "../type.h"

class CoreMatrix
{
public:
	CoreMatrix();
	CoreMatrix(uint row, uint column);

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
