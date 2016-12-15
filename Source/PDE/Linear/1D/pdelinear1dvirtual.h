#ifndef PDELINEAR1DVIRTUAL_H
#define PDELINEAR1DVIRTUAL_H

#include "../pdelinearvirtual.h"

class PDELinear1DVirtual: public PDELinearVirtual
{
public:
	PDELinear1DVirtual();
	PDELinear1DVirtual(const Frame &F, int Past, int Future);
	virtual ~PDELinear1DVirtual();

	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;
};

#endif // PDELINEAR1DVIRTUAL_H
