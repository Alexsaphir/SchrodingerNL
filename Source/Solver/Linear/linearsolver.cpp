#include "linearsolver.h"

LinearSolver::LinearSolver(): Solver()
{
	System = new SparseMatrix(10,10);
}

LinearSolver::LinearSolver(uint size)
{
	System = new SparseMatrix(size,size);
}


void LinearSolver::initSolver(Type dt, Type dx)
{
	if(!System)
		return;
	Type alpha=dt/dx/dx;
	System->setValue(0, 0, 1.);
	for(uint i(1); i<(System->row()-1); ++i)
	{
		System->setValue(i, i, 1.+2.*alpha);
		System->setValue(i, i-1, -alpha);//uint => 0-1==2^32
		System->setValue(i, i+1, -alpha);//if out of order, it's catch by the sparseMatrix
	}
	System->setValue(System->row()-1, System->row()-1, 1.);
}

void LinearSolver::SORMethod(const ColumnMatrixVirtual *B, ColumnMatrixVirtual *X)
{
	uint n(System->row());
	Type w(1.5);
	//set X to 0
	for(uint i(0); i<n; ++i)
		X->set(i,0);

	int step(0);
	while(step!=50)
	{
		for(uint i=0;i<n;++i)
		{
			cmplx sigma(0,0);
			for(uint j=0;j<n;++j)
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


LinearSolver::~LinearSolver()
{
	delete System;
}
