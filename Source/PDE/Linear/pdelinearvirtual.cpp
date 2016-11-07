#include "pdelinearvirtual.h"

PDELinearVirtual::PDELinearVirtual(): PDEVirtual()
{
	LS = NULL;
}

PDELinearVirtual::PDELinearVirtual(const Frame &F): PDEVirtual(F)
{
	LS = new LinearSolver(F.size());
}

PDELinearVirtual::PDELinearVirtual(const Frame &F, int Past, int Future, cmplx BoundExt): PDEVirtual(F, Past, Future, BoundExt)
{
	LS = new LinearSolver(F.size());
}

PDELinearVirtual::~PDELinearVirtual()
{
	if(LS)
		delete LS;
}
