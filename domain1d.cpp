#include "domain1d.h"

Domain1D::Domain1D(Type Xmin, Type Xmax, Type Xstep, cmplx Binf, cmplx Bsup) :Grid1D(Xmin, Xmax, Xstep)
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
//	Reflective boundary
//	if(i==-1)
//		return Grid1D::getValue(0);
//	if(i==this->getN())
//		return Grid1D::getValue(this->getN()-1);


//	Constant boundary
	if(i<0)
		return BoundInf;
	if(i==this->getN())
		return BoundSup;

	return Grid1D::getValue(i);
}

void Domain1D::doFourrier()
{
	Grid1D Tmp= *this;

	for (int k=0;k<this->getN();++k)
	{
		cmplx i(0,-2.*M_PI*k/this->getN());
		cmplx v(0,0);
		for (int n=0;n<this->getN();++n)
		{
			v+=Tmp.getValue(n)*std::exp((Type)n*i);
		}
		this->setValue(k,v);
	}
}

void Domain1D::undoFourrier()
{
	Grid1D Tmp= *this;

	for (int n=0;n<this->getN();++n)
	{
		cmplx j(0.,2.*M_PI*(Type)n/(Type)this->getN());

		cmplx v(0,0);
		for (int k=0;k<this->getN();++k)
		{
			v+=Tmp.getValue(k)*std::exp(j*(Type)k);
		}
		this->setValue(n,v/(Type)this->getN());
	}
}
