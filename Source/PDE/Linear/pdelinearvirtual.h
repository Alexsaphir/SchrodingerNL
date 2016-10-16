#ifndef PDELINEARVIRTUAL_H
#define PDELINEARVIRTUAL_H

#include "../pdevirtual.h"
#include "../../Solver/Linear/linearsolver.h"

class PDELinearVirtual: PDEVirtual
{
public:
	PDELinearVirtual();
	PDELinearVirtual(uint LS_size);

	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;

	virtual ~PDELinearVirtual();
public:
	//LinearSolver *LS;
};

#endif // PDELINEARVIRTUAL_H
