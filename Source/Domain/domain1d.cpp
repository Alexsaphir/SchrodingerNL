#include "domain1d.h"

Domain1D::Domain1D(const Axis *X, cmplx Binf, cmplx Bsup) : Domain(Frame(X),Binf)
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

	return Grid::getValue(i);
}

Domain1D::~Domain1D()
{

}
