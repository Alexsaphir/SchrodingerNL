#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "../solver.h"
#include "../../Matrix/SparseMatrix/sparsematrix.h"

class LinearSolver: public Solver
{
public:
	LinearSolver();
	LinearSolver(SparseMatrix *M);
	virtual void initSolver();

	virtual ~LinearSolver();
private:
	SparseMatrix *System;

};

#endif // LINEARSOLVER_H
