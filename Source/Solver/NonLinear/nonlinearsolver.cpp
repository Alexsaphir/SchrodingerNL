#include "nonlinearsolver.h"

NonLinearSolver::NonLinearSolver()
{
	Systeme = new SparseMatrix(0, 0);
}

NonLinearSolver::NonLinearSolver(int sizeSystem)
{
	if(sizeSystem<0)
		sizeSystem = 0;
	Systeme = new SparseMatrix(sizeSystem, sizeSystem);
}

SparseMatrix* NonLinearSolver::getSysteme() const
{
	return Systeme;
}

void NonLinearSolver::GaussSeidel() const
{

}

NonLinearSolver::~NonLinearSolver()
{
	delete Systeme;
}
