#include "include/domain2d.h"

Domain2D::Domain2D(const Axis &X, const Axis &Y, cmplx Bext) : Grid2D(X, Y)
{
	BoundExt = Bext;
}

cmplx Domain2D::getValue(int i, int j) const
{
	if(isInGrid(i, j))
		return Grid2D::getValue(i, j);
	else
	{
		//Boundary conditions

		return BoundExt;
	}
}
