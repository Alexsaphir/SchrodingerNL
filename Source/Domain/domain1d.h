#ifndef DOMAIN1D_H
#define DOMAIN1D_H

#include "domain.h"



class Domain1D: public Domain
{
public:
	Domain1D(const Axis *X, cmplx Binf, cmplx Bsup);
	~Domain1D();

	cmplx getValue(int i) const;

private:
	cmplx BoundInf;//getValue(-1)
	cmplx BoundSup;//getValue(N+1)
};



#endif // DOMAIN1D_H
