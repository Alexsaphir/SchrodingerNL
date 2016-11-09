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
	//set X to 0
	for(int i(0); i<n; ++i)
		X->set(i,0);

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

void LinearSolver::SORMethod(const Grid1D *B, Grid1D *X)
{
	int n(System->row());
	Type w(.5);
	//set X to 0
	for(int i(1); i<n-1; ++i)
		X->setValue(i,0);

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
					sigma+=System->getValue(i,j)*X->getValue(j);
				}
			}
			cmplx Res=B->getValue(i)-sigma;
			Res/= System->getValue(i,i);
			Res-=X->getValue(i);
			Res*=w;
			Res+=X->getValue(i);
			X->setValue(i, Res);
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
