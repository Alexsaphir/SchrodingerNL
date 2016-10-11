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
	SparseMatrix(uint row, uint column);

	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);

	void dotByGrid1D(Grid1D *S, Grid1D *R);

	virtual ~SparseMatrix();
private:
	QVector<QMap<uint,cmplx>*> V;

};

#endif // SPARSEMATRIX_H
