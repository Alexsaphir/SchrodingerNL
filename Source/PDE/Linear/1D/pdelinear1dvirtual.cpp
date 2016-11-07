#include "pdelinear1dvirtual.h"

PDELinear1DVirtual::PDELinear1DVirtual(): PDELinearVirtual()
{

}

PDELinear1DVirtual::PDELinear1DVirtual(const Frame &F, int Past, int Future, cmplx BoundExt): PDELinearVirtual(F, Past, Future, BoundExt)
{

}


PDELinear1DVirtual::~PDELinear1DVirtual()
{

}
