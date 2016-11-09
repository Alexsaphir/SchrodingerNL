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
	~RowDataProxy();

	void setDomain(Domain *D);

	//Specific method for Row or Column Matrix
	virtual cmplx at(int i) const;
	virtual void set(int i, const cmplx &value);

	//Generic method
	virtual cmplx getValue(int i, int j) const;
	virtual void setValue(int i, int j, const cmplx &value);

	virtual int row() const;
	virtual int column() const;

private:
	Domain *m_domain;
};

#endif // ROWDATAPROXY_H
