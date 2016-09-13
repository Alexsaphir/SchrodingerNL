#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "domain1d.h"


class Integration
{
public:
	Integration();
	static Type integrate(Domain1D const &D);
	static Type integrate(Domain1D const *D);
};

#endif // INTEGRATION_H
