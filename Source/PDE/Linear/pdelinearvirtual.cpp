#include "pdelinearvirtual.h"

PDELinearVirtual::PDELinearVirtual(): PDEVirtual()
{
	LS = NULL;
}

PDELinearVirtual::PDELinearVirtual(const Frame &F): PDEVirtual(F)
{
	LS = new LinearSolver(F.size());
}

PDELinearVirtual::PDELinearVirtual(const Frame &F, int Past, int Future): PDEVirtual(F, Past, Future)
{
	//We need to compute the number of points in Frame
	LS = new LinearSolver(m_Space->getCurrentGrid()->getSizeOfGrid());
}

PDELinearVirtual::~PDELinearVirtual()
{
	if(LS)
		delete LS;
}
