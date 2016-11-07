#include "../Grid/gridmanager.h"

GridManager::GridManager(const Axis *X, cmplx Binf, cmplx Bsup, int i, int d)
{
	if(i<=0)
		i=1;
	if(d>i)
		d=0;

	for(int j=0;j<i;++j)
	{
		Domain1D* tmp = new Domain1D(X, Binf, Bsup);
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
	//Current == 0
	//Next == 1
	//Old == -1

	if( (i+offset)<Stack.size() && (i+offset)>=0)
		return Stack.at(i+offset);
	else
		return getCurrentDomain();
}

Domain1D* GridManager::getNextDomain() const
{
	return getDomain(1);
}

Domain1D* GridManager::getOldDomain() const
{
	return getDomain(-1);
}

int GridManager::getSizeStack() const
{
	return Stack.size();
}

void GridManager::switchDomain()
{
	Domain1D *begin;
	begin = Stack.first();
	for(int i=0;i<Stack.size()-1;++i)
	{
		Stack.replace(i,Stack.at(i+1));
	}
	Stack.replace(Stack.size()-1, begin);
}

GridManager::~GridManager()
{
	while(!Stack.empty())
	{
		delete Stack.first();
		Stack.removeFirst();
	}
}
