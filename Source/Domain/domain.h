#ifndef DOMAIN_H
#define DOMAIN_H

#include "../Grid/grid.h"

class Domain: public Grid
{
public:
	Domain();
	Domain(cmplx Bext);
	Domain(const Domain &D);

	cmplx getValue(const Point &Pos) const;
	cmplx getValue(int i) const;
private:
	cmplx getBoundaryCondition(const Point &Pos) const;
private:
	cmplx BoundExt;
};

#endif // DOMAIN_H
