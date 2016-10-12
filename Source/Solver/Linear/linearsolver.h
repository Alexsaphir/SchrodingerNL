#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "../solver.h"
#include "../../Matrix/SparseMatrix/sparsematrix.h"
#include "../../Matrix/Matrix/ColumnMatrix/columnmatrixvirtual.h"

class LinearSolver: public Solver
{
public:
	LinearSolver();
	LinearSolver(uint size);

	virtual void initSolver(Type dt, Type dx);

	void SORMethod(const ColumnMatrixVirtual *B, ColumnMatrixVirtual *X);

	virtual ~LinearSolver();
private:
	SparseMatrix *System;

};

#endif // LINEARSOLVER_H
