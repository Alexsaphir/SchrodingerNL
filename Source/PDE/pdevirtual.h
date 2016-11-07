#ifndef PDEVIRTUAL_H
#define PDEVIRTUAL_H


#include "../frame.h"
#include "../point.h"
#include "../type.h"
#include "../Domain/domainmanager.h"

class PDEVirtual
{
public:
	PDEVirtual();
	PDEVirtual(const Frame &F);
	PDEVirtual(const Frame &F, int Past, int Future, cmplx BoundExt);

	virtual void computeNextStep() =0;
	virtual void InitialState() = 0;

	virtual cmplx at(const Point &P) const;

	virtual ~PDEVirtual();

public:
	Frame *Repere;
	DomainManager *Space;

};

#endif // PDEVIRTUAL_H
