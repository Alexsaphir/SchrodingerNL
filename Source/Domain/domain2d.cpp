#include "domain2d.h"

Domain2D::Domain2D(const Axis *X, const Axis *Y, cmplx Bext) : Domain(X, Y, Bext)
{
}

cmplx Domain2D::getValue(int i, int j) const
{
	if(Domain::isInGrid(Point(i,j)))
		return GridPrivate::getValue(Point(i,j));
	else
	{
		//Boundary conditions
		return m_BoundExt;
	}
}

Domain2D::~Domain2D()
{

}
