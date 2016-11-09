#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "../solver.h"
#include "../../Matrix/SparseMatrix/sparsematrix.h"
#include "../../Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.h"

class LinearSolver: public Solver
{
public:
	LinearSolver();
	LinearSolver(int size);
	virtual ~LinearSolver();

	virtual void initSolver();
	SparseMatrix* getSparseMatrix() const;

	void SORMethod(const ColumnMatrixVirtual *B, ColumnMatrixVirtual *X);


private:
	SparseMatrix *System;

};

#endif // LINEARSOLVER_H
