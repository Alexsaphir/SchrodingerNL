#ifndef NONLINEARSOLVER_H
#define NONLINEARSOLVER_H

#include "../../type.h"
#include "../../Matrix/SparseMatrix/sparsematrix.h"

class NonLinearSolver
{
public:
	NonLinearSolver();
	NonLinearSolver(int sizeSystem);
	~NonLinearSolver();

	SparseMatrix* getSysteme() const;
	void GaussSeidel() const;

protected:
	SparseMatrix *Systeme;
};

#endif // NONLINEARSOLVER_H
