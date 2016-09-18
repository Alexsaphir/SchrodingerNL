#include "include/integration.h"

Integration::Integration()
{

}

Type Integration::integrate(const Domain1D &D)
{
	Type sum(0.);
	for(int i=0;i<D.getN();++i)
	{
		sum+=D.getDx()*std::abs(D.getValue(i));
	}
	return sum;
}

Type Integration::integrate(const Domain1D *D)
{
	Type sum(0.);

	for(int i=0;i<D->getN();++i)
	{
		sum+=D->getDx()*std::abs(D->getValue(i));
	}
	return sum;
}
