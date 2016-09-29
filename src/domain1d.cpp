#include "include/domain1d.h"

Domain1D::Domain1D(const Axis &X, cmplx Binf, cmplx Bsup) : Grid1D(X)
{
	BoundInf = Binf;
	BoundSup = Bsup;
}


cmplx Domain1D::getValue(int i) const
{
//	Constant boundary
	if(i<0)
		return BoundInf;
	if(i>=this->getN())
		return BoundSup;

	return Grid1D::getValue(i);
}

void Domain1D::doFourrier()
{
	Grid1D Tmp= *this;
#pragma omp parallel for
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
#pragma omp parallel for
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
