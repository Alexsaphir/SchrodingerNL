#include "heat1d.h"

Heat1D::Heat1D(): PDELinear1DVirtual(), Grid1(NULL), Grid2(NULL), Grid1IsCurrent(true)
{

}
Heat1D::Heat1D(const Axis &X, Type t, cmplx Binf, cmplx Bsup): PDELinear1DVirtual()
{

}


Heat1D::~Heat1D()
{
	if(Grid1)
		delete Grid1;
	if(Grid2)
		delete Grid2;
}
