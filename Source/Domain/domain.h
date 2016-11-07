#ifndef DOMAIN_H
#define DOMAIN_H

#include "../Grid/grid.h"

class Domain: public Grid
{
public:
	Domain();
	Domain(const Frame &F, cmplx Bext);
	Domain(const Axis *X, cmplx Bext);
	Domain(const Axis *X, const Axis *Y, cmplx Bext);
	Domain(const Domain &D);

	cmplx getValue(const Point &Pos) const;
	virtual cmplx getValue(int i) const;

	virtual ~Domain();
private:
	cmplx getBoundaryCondition(const Point &Pos) const;
protected:
	cmplx BoundExt;
};

#endif // DOMAIN_H
