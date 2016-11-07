#include "pdevirtual.h"

PDEVirtual::PDEVirtual()
{
	Repere = NULL;
	Space = NULL;
}

PDEVirtual::PDEVirtual(const Frame &F)
{
	Repere = new Frame(F);
	Space = new DomainManager(0, 0, *Repere, 0.);
}

PDEVirtual::PDEVirtual(const Frame &F, int Past, int Future, cmplx BoundExt)
{
	Repere = new Frame(F);
	Space = new DomainManager(Past, Future, *Repere, BoundExt);
}

cmplx PDEVirtual::at(const Point &P) const
{
	return cmplx(0,0);
}

PDEVirtual::~PDEVirtual()
{
	delete Repere;
	delete Space;
}
