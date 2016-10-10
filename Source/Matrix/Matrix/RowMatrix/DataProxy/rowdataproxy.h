#ifndef ROWDATAPROXY_H
#define ROWDATAPROXY_H

#include "../rowmatrixvirtual.h"
#include "../../../../Domain/domain.h"
#include "../../../../type.h"

class RowDataProxy: public RowMatrixVirtual
{
public:
	RowDataProxy();
	RowDataProxy(Domain *D);

	void setDomain(Domain *D);

	//Specific method for Row or Column Matrix
	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);
	//Generic method
	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);

	virtual uint row() const;
	virtual uint column() const;

	~RowDataProxy();
private:
	Domain *m_domain;
};

#endif // ROWDATAPROXY_H
