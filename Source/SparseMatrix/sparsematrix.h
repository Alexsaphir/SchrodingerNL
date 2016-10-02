#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <QPair>
#include <QMap>

#include "../type.h"

class SparseMatrix
{
public:
	SparseMatrix();
	SparseMatrix(int i,int j);

	cmplx getValue(const Position &P) const;
	void setValue(const Position &P, const cmplx &v);


private:
	QMap<Position,cmplx> Map;
	int row;
	int column;

};

#endif // SPARSEMATRIX_H
