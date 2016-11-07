#ifndef PDELINEARVIRTUAL_H
#define PDELINEARVIRTUAL_H

#include "../pdevirtual.h"
#include "../../Solver/Linear/linearsolver.h"

class PDELinearVirtual: public PDEVirtual
{
public:
	PDELinearVirtual();
	PDELinearVirtual(const Frame &F);
	PDELinearVirtual(const Frame &F, int Past, int Future, cmplx BoundExt);

	virtual void computeNextStep()			=0;
	virtual void InitialState()				= 0;
	virtual void initializeLinearSolver()	= 0;
	virtual ~PDELinearVirtual();
protected:
	LinearSolver *LS;

};

#endif // PDELINEARVIRTUAL_H
