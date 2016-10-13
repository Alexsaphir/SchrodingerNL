#ifndef PDELINEAR1DVIRTUAL_H
#define PDELINEAR1DVIRTUAL_H

#include "../pdelinearvirtual.h"

class PDELinear1DVirtual: PDELinearVirtual
{
public:
	PDELinear1DVirtual();
	PDELinear1DVirtual(uint LS_size);

	virtual void initMatrix()=0;
	virtual void pulse()=0;
	virtual void compute()=0;
	virtual cmplx get(uint i) const=0;

	virtual ~PDELinear1DVirtual();
};

#endif // PDELINEAR1DVIRTUAL_H
