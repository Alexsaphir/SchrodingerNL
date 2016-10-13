#ifndef PDELINEARVIRTUAL_H
#define PDELINEARVIRTUAL_H

#include "../pdevirtual.h"
#include "../../Solver/Linear/linearsolver.h"

class PDELinearVirtual: PDEVirtual
{
public:
	PDELinearVirtual();
	PDELinearVirtual(uint LS_size);

	virtual void initMatrix()=0;
	virtual void pulse()=0;
	virtual void compute()=0;
	virtual cmplx get(uint i) const=0;

	virtual ~PDELinearVirtual();
public:
	//LinearSolver *LS;
};

#endif // PDELINEARVIRTUAL_H
