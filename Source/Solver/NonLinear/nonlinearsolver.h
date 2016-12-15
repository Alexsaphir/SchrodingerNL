#ifndef NONLINEARSOLVER_H
#define NONLINEARSOLVER_H

#include "../../type.h"
#include "../../Matrix/SparseMatrix/sparsematrix.h"
#include "../../Systeme/systemevirtual.h"

class NonLinearSolver
{
public:
	NonLinearSolver();
	NonLinearSolver(int sizeSystem);
	~NonLinearSolver();

	void Newton(SystemeVirtual *S, GridBase *Result, GridBase *InitialGuess) const;
};

#endif // NONLINEARSOLVER_H
