#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <QPair>
#include <QMap>

#include <QDebug>

#include "../../type.h"
#include "../corematrix.h"
#include "../../Grid/grid1d.h"

class SparseMatrix: public CoreMatrix
{
public:
	SparseMatrix(int row, int column);
	~SparseMatrix();
	cmplx getValue(int i, int j) const;
	void setValue(int i, int j, const cmplx &value);

	void dotByGrid1D(Grid1D *S, Grid1D *R);


private:
	QVector<QMap<int,cmplx>*> V;
};

#endif // SPARSEMATRIX_H
