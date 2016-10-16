#ifndef PDEVIRTUAL_H
#define PDEVIRTUAL_H

#include "../type.h"

class PDEVirtual
{
public:
	PDEVirtual();

	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;

	virtual ~PDEVirtual();
};

#endif // PDEVIRTUAL_H
