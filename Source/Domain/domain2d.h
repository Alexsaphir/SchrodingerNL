#ifndef DOMAIN2D_H
#define DOMAIN2D_H

#include "domain.h"

class Domain2D : public Domain
{
public:
	Domain2D(const Axis *X, const Axis *Y, cmplx Bext);
	~Domain2D();

	cmplx getValue(int i, int j) const;
};

#endif // DOMAIN2D_H
