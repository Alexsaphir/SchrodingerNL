#ifndef DOMAINMANAGERBASE_H
#define DOMAINMANAGERBASE_H

#include <QList>

#include "domainbase.h"
#include "../../frame.h"
#include "../../Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"
#include "../../Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.h"

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

	cmplx getValue(const Point &P, int t) const;

	void switchDomain();

	ColumnDataProxy* getCurrentColumn() const;//Old
	ColumnDataProxy* getNextColumn() const;//Old

	ColumnDataProxy* getColumnAtTime(int t) const;
	RowDataProxy* getRowAtTime(int t) const;

protected:
	const Frame *m_Frame;
	int m_Size;//Number of DomainBase
	int m_offset;//Indice of the Current Domain

	QList<DomainBase*> m_Stack;
	QList<ColumnDataProxy*> m_ProxyColumn;
	QList<RowDataProxy*> m_ProxyRow;
};

#endif // DOMAINMANAGERBASE_H
