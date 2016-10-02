#ifndef DOMAIN1D_H
#define DOMAIN1D_H

#include "../Grid/grid1d.h"



class Domain1D: public Grid1D
{
public:
	Domain1D(const Axis &X, cmplx Binf, cmplx Bsup);

	cmplx getValue(int i) const;

	void doFourrier();
	void undoFourrier();
private:
	cmplx BoundInf;//getValue(-1)
	cmplx BoundSup;//getValue(N+1)
};



#endif // DOMAIN1D_H
