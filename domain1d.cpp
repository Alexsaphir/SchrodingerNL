#include "domain1d.h"

Domain1D::Domain1D(double Xmin, double Xmax, double Xstep, cmplx Binf, cmplx Bsup) :Grid1D(Xmin, Xmax, Xstep)
{
	BoundInf = Binf;
	BoundSup = Bsup;
}

void Domain1D::setValue(int i,cmplx y)
{
	Grid1D::setValue(i,y);
}

cmplx Domain1D::getValue(int i) const
{
//	if(i==-1)
//		return Grid1D::getValue(0);
//	if(i==this->getN())
//		return Grid1D::getValue(this->getN()-1);
	//Borne constante
	if(i==-1)
		return BoundInf;
	if(i==this->getN())
		return BoundSup;
	return Grid1D::getValue(i);
}
