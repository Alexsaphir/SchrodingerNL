#ifndef MATRIXALGORITHM_H
#define MATRIXALGORITHM_H

#include "../corematrix.h"
#include "../Matrix/matrix.h"

#include "../Matrix/ColumnMatrix/columnmatrix.h"
#include "../Matrix/ColumnMatrix/columnmatrixvirtual.h"
#include "../Matrix/ColumnMatrix/DataProxy/columndataproxy.h"

#include "../Matrix/RowMatrix/rowmatrix.h"
#include "../Matrix/RowMatrix/rowmatrixvirtual.h"
#include "../Matrix/RowMatrix/DataProxy/rowdataproxy.h"

#include "../SparseMatrix/sparsematrix.h"

class MatrixAlgorithm
{
public:
	MatrixAlgorithm();

	void MatrixAddition(const CoreMatrix *A, const CoreMatrix *B, CoreMatrix *C);//A+B=C
};

#endif // MATRIXALGORITHM_H
