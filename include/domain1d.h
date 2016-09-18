#ifndef DOMAIN1D_H
#define DOMAIN1D_H

#include "grid1d.h"



class Domain1D: public Grid1D
{
public:
	Domain1D(Type Xmin, Type Xmax, Type Xstep, cmplx Binf, cmplx Bsup);

	cmplx getValue(int i) const;
	void setValue(int i, cmplx y);

	void doFourrier();
	void undoFourrier();
private:
	cmplx BoundInf;//getValue(-1)
	cmplx BoundSup;//getValue(N+1)
};



#endif // DOMAIN1D_H
