#ifndef DOMAINMANAGERBASE_H
#define DOMAINMANAGERBASE_H

#include <QList>

#include "domainbase.h"
#include "../../frame.h"

class DomainManagerBase
{
public:
	DomainManagerBase();
	DomainManagerBase(int PastDomain, int FutureDomain, const Frame *F, cmplx BoundExt);
	~DomainManagerBase();

	int getSizeStack();

	DomainBase* getDomain(int i) const;
	DomainBase* getCurrentDomain() const;
	DomainBase* getNextDomain() const;
	DomainBase* getOldDomain() const;

	void switchDomain();

protected:
	const Frame *m_Frame;
	int m_Size;//Number of DomainBase
	int m_offset;//Indice of the Current Domain
	QList<DomainBase*> m_Stack;
};

#endif // DOMAINMANAGERBASE_H
