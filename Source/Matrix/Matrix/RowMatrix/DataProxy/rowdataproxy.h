#ifndef ROWDATAPROXY_H
#define ROWDATAPROXY_H

#include "../rowmatrixvirtual.h"
#include "../../../../Domain/Base/domainbase.h"
#include "../../../../type.h"

class RowDataProxy: public RowMatrixVirtual
{
public:
	RowDataProxy();
	RowDataProxy(DomainBase *D);
	~RowDataProxy();

	void setDomain(DomainBase *D);

	//Specific method for Row or Column Matrix
	cmplx at(int i) const;
	void set(int i, const cmplx &value);

	//Generic method
	cmplx getValue(int i, int j) const;
	void setValue(int i, int j, const cmplx &value);

	int row() const;
	int column() const;

private:
	DomainBase *m_domain;
};

#endif // ROWDATAPROXY_H
