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

	virtual void initSolver(Type dt, Type dx);
	SparseMatrix* getSparseMatrix() const;

	void SORMethod(const ColumnMatrixVirtual *B, ColumnMatrixVirtual *X);
	void SORMethod(const Grid1D *B, Grid1D *X);

	virtual ~LinearSolver();
private:
	SparseMatrix *System;

};

#endif // LINEARSOLVER_H
