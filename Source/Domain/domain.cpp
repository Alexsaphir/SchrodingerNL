#include "domain.h"

Domain::Domain() : Grid()
{

}

Domain::Domain(const Domain &D) : Grid(D)
{
	BoundExt = D.BoundExt;
}

Domain::Domain(cmplx Bext) : Grid(), BoundExt(Bext)
{

}

cmplx Domain::getValue(const Point &Pos) const
{
	int i(getIndexFromPos(Pos));
	if(i == -1)
		return getBoundaryCondition(Pos);
	return Grid::getValue(i);
}

cmplx Domain::getValue(int i) const
{
	//We can't use many boundary condition
	if(i<0 || i>=getN())
		return BoundExt;
	return Grid::getValue(i);
}

cmplx Domain::getBoundaryCondition(const Point &Pos) const
{
	return BoundExt;
}
