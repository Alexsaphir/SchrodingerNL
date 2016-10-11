#include "linearsolver.h"

LinearSolver::LinearSolver(): Solver()
{
	System = new SparseMatrix(10,10);
}

LinearSolver::LinearSolver(SparseMatrix *M)
{
	System = M;
}

void LinearSolver::initSolver()
{

}


LinearSolver::~LinearSolver()
{
	delete System;
}
