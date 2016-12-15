#include "nonlinearsolver.h"

NonLinearSolver::NonLinearSolver()
{
}

NonLinearSolver::NonLinearSolver(int sizeSystem)
{
}

void NonLinearSolver::Newton(SystemeVirtual *S, GridBase *Result, GridBase *InitialGuess) const
{
	GridBase *B=new GridBase(*InitialGuess);

	//compute B=-f(InitialGuess)
	S->evaluate(InitialGuess, B);
	for(int i=0; i<B->getSizeOfGrid(); ++i)
	{
		B->setValue(i, -B->getValue(i));
		Result->setValue(i, InitialGuess->getValue(i));//Copy InitialGuess in Result
	}

	S->computeJacobian(Result);
}

NonLinearSolver::~NonLinearSolver()
{
}
