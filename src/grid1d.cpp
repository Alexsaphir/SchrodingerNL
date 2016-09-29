#include "include/grid1d.h"


Grid1D::Grid1D(const Axis &X)
{
	AxisX = X;

	V.fill(cmplx(0.,0.),getN());
}

Type Grid1D::getDx() const
{
	return AxisX.getAxisStep();
}

Type Grid1D::getPos(int i) const
{
	if(i<0)
		return 0.;
	if(i>=this->getN())
		return 0.;
	return getXmin()+getDx()*(double)(i);
}

cmplx Grid1D::getValue(int i) const
{
	//It's the Domain class who catch the index error
	return V.at(i);
}

Type Grid1D::getXmax() const
{
	return AxisX.getAxisMax();
}

Type Grid1D::getXmin() const
{
	return AxisX.getAxisMin();
}

int Grid1D::getN() const
{
	return AxisX.getAxisStep();
}

void Grid1D::setValue(int i, cmplx value)
{
	if(i<0)
		return;
	if(i>this->getN())
		return;

	V.replace(i, value);
}

