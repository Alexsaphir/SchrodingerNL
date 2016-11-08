#include "domainmanager.h"

DomainManager::DomainManager(int PastDomain, int FutureDomain, const Frame &F, cmplx Bext)
{
	if(PastDomain<0)
		PastDomain = 0;
	if(FutureDomain<0)
		FutureDomain = 0;


	for(int i=0; i<(PastDomain + FutureDomain + 1); ++i)
	{
		Domain *tmp = new Domain(F, Bext);
		Stack.append(tmp);
	}
	offset = PastDomain;
	Size = Stack.size();

	CurrProxy = new ColumnDataProxy(getCurrentDomain());
	NextProxy = new ColumnDataProxy(getNextDomain());
}


Domain* DomainManager::getCurrentDomain() const
{
	return Stack.at(offset);
}

Domain* DomainManager::getDomain(int i) const
{
	//Current == 0
	//Next == 1
	//Old == -1
	if( (offset+i)<0 || (offset+i)>Size)
		return Stack.at(offset);

	return Stack.at(offset+i);
}

Domain* DomainManager::getNextDomain() const
{
	return getDomain(1);
}

Domain* DomainManager::getOldDomain() const
{
	return getDomain(-1);
}

int DomainManager::getSizeStack() const
{
	return Size;
}

void DomainManager::switchDomain()
{
	Domain *begin;
	begin = Stack.first();
	Stack.removeFirst();
	Stack.append(begin);
}

ColumnDataProxy* DomainManager::getCurrentColumn() const
{
	return CurrProxy;
}

ColumnDataProxy* DomainManager::getNextColumn() const
{
	return NextProxy;
}

DomainManager::~DomainManager()
{
	while(!Stack.empty())
	{
		delete Stack.first();
		Stack.removeFirst();
	}
}
