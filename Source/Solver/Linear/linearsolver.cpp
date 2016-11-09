#include "linearsolver.h"

LinearSolver::LinearSolver(): Solver()
{
	System = new SparseMatrix(10,10);
}

LinearSolver::LinearSolver(int size)
{
	System = new SparseMatrix(size,size);
}


void LinearSolver::initSolver()
{
}

void LinearSolver::SORMethod(const ColumnMatrixVirtual *B, ColumnMatrixVirtual *X)
{
	int n(System->row());
	Type w(.5);
	//set X to 0, it's a bad idea
	for(int i(0); i<n; ++i)
		X->set(i,B->at(i));

	int step(0);
	while(step!=50)
	{
		for(int i=0;i<n;++i)
		{
			cmplx sigma(0,0);
			for(int j=0;j<n;++j)
			{
				if (j!=i)
				{
					sigma+=System->getValue(i,j)*X->at(j);
				}
			}
			cmplx Res=B->at(i)-sigma;
			Res/= System->getValue(i,i);
			Res-=X->at(i);
			Res*=w;
			Res+=X->at(i);
			X->set(i, Res);
		}
		++step;
	}
}

SparseMatrix* LinearSolver::getSparseMatrix() const
{
	return System;
}

LinearSolver::~LinearSolver()
{
	delete System;
}
