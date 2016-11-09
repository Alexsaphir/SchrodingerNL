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
	//We need to compute the number of points in Frame
	LS = new LinearSolver(m_Space->getCurrentDomain()->getSizeOfGrid());
}

PDELinearVirtual::~PDELinearVirtual()
{
	if(LS)
		delete LS;
}
