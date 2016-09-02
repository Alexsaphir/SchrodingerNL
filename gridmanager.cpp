#include "gridmanager.h"

GridManager::GridManager(double Xmin, double Xmax, double Xstep, cmplx Binf, cmplx Bsup, int i, int d)
{
	if(i<=0)
		i=1;
	if(d>i)
		d=0;

	for(int j=0;j<i;++j)
	{
		Domain1D* tmp = new Domain1D(Xmin, Xmax, Xstep, Binf, Bsup);
		Stack.append(tmp);
	}
	offset = d;
}

Domain1D* GridManager::getCurrentDomain() const
{
	return Stack.at(offset);
}

Domain1D* GridManager::getDomain(int i) const
{
	if( (i+offset)<Stack.size() )
		return Stack.at(i+offset);
	else
		return getCurrentDomain();
}

Domain1D* GridManager::getNextDomain() const
{
	return getDomain(1);
}

int GridManager::getSizeStack() const
{
	return Stack.size();
}

GridManager::~GridManager()
{
	while(!Stack.empty())
	{
		delete Stack.first();
		Stack.removeFirst();
	}
}
