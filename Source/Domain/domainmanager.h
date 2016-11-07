#ifndef DOMAINMANAGER_H
#define DOMAINMANAGER_H

#include <QList>

#include "domain.h"


class DomainManager
{
public:
	DomainManager(int PastDomain, int FutureDomain, const Frame &F, cmplx Bext);

	int getSizeStack() const;

	Domain* getDomain(int i) const;
	Domain* getCurrentDomain() const;
	Domain* getNextDomain() const;
	Domain* getOldDomain() const;

	void switchDomain();

	~DomainManager();

private:
	QList<Domain*> Stack;
	int Size;
	int offset;//Indice of the Current Domain
	bool isInit;
};

#endif // DOMAINMANAGER_H
