#ifndef DOMAIN2D_H
#define DOMAIN2D_H

#include "../Grid/grid2d.h"

class Domain2D : public Grid2D
{
public:
	Domain2D(const Axis &X, const Axis &Y, cmplx Bext);

	cmplx getValue(int i, int j) const;

private:
	cmplx BoundExt;//In 2D boundary condition are more difficult than in 1D
};

#endif // DOMAIN2D_H
