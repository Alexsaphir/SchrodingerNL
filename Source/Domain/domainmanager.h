#ifndef DOMAINMANAGER_H
#define DOMAINMANAGER_H

#include <QList>

#include "Base/domainbase.h"
#include "Base/domainmanagerbase.h"

class DomainManager : public DomainManagerBase
{
public:
	DomainManager(int PastDomain, int FutureDomain, const Frame &F, cmplx BoundExt);
	~DomainManager();

	ColumnDataProxy *getCurrentColumn() const;
	ColumnDataProxy *getNextColumn() const;

private:
	const Frame *m_Frame;
};

#endif // DOMAINMANAGER_H
