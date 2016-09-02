#include "grid1d.h"


Grid1D::Grid1D(double Xmn, double Xmx, double Xsp)
{
	Xmin = Xmn;
	Xmax = Xmx;
	Xstep = Xsp;

	nbPts = (int)((-Xmin + Xmax)/Xstep)+1;

	V.fill(0.,nbPts);
}

double Grid1D::getDx() const
{
	return Xstep;
}

double Grid1D::getPos(int i) const
{
	if(i<0)
		return 0.;
	if(i>=this->getN())
		return 0.;
	return Xmin+Xstep*(double)(i);
}

cmplx Grid1D::getValue(int i) const
{
	if(i<0)
		return 0.;
	if(i>=this->getN())
		return 0.;
	return V.at(i);
}

double Grid1D::getXmax() const
{
	return Xmax;
}

double Grid1D::getXmin() const
{
	return Xmin;
}

int Grid1D::getN() const
{
	return nbPts;
}

void Grid1D::setValue(int i, cmplx y)
{
	if(i<0)
		return;
	if(i>this->getN())
		return;

	V.replace(i, y);
}

void Grid1D::setValueReal(int i, Type y)
{
	if(i<0)
		return;
	if(i>this->getN())
		return;
	cmplx tmp(y, V.at(i).imag());
	V.replace(i, tmp);
}

void Grid1D::setValueImag(int i, Type y)
{
	if(i<0)
		return;
	if(i>this->getN())
		return;
	cmplx tmp(V.at(i).real(), y);
	V.replace(i, tmp);
}
