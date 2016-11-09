#include "grid2d.h"

Grid2D::Grid2D(const Axis *X, const Axis *Y)
{
	AxisX = X->clone();
	AxisY = Y->clone();
	V.fill(cmplx(0.,0.),getNx()*getDy());
}

Type Grid2D::getDx() const
{
	return AxisX->getAxisStep();
}

Type Grid2D::getDy() const
{
	return AxisY->getAxisStep();
}

int Grid2D::getNx() const
{
	return AxisX->getAxisN();
}

int Grid2D::getNy() const
{
	return AxisY->getAxisN();
}

Type Grid2D::getPosX(int i, int j) const
{
	if (isInGrid(i, j))
		return getDx()*(Type)i;
	else
		return (Type)0;
}

Type Grid2D::getPosY(int i, int j) const
{
	if (isInGrid(i, j))
		return getDy()*(Type)j;
	else
		return (Type)0;
}

cmplx Grid2D::getValue(int i, int j) const
{
	//The domain look if (i,j) is in the grid or give a boundary condition
	//We can skip the test if (i,j) is in the grid

	return V.at(j+getNy()*i);
}

Type Grid2D::getXmax() const
{
	return AxisX->getAxisMax();
}

Type Grid2D::getXmin() const
{
	return AxisX->getAxisMin();
}

Type Grid2D::getYmax() const
{
	return AxisY->getAxisMax();
}

Type Grid2D::getYmin() const
{
	return AxisY->getAxisMin();
}

bool Grid2D::isInGrid(int i, int j) const
{
	if(i<0 || i>=getNx() || j<0 || j>=getNy())
		return false;
	else
		return true;
}

void Grid2D::setValue(int i, int j, cmplx value)
{
	if (!isInGrid(i,j))
		return;
	V.replace(j+getNy()*i, value);
}
