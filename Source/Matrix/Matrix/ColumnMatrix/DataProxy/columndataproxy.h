#ifndef COLUMNDATAPROXY_H
#define COLUMNDATAPROXY_H

#include "../columnmatrixvirtual.h"
#include "../../../../Domain/domain.h"
#include "../../../../type.h"

class ColumnDataProxy: public ColumnMatrixVirtual
{
public:
	ColumnDataProxy();
	ColumnDataProxy(Domain *D);

	void setDomain(Domain *D);

	//Specific method for Row or Column Matrix
	virtual cmplx at(uint i) const;
	virtual void set(uint i, const cmplx &value);
	//Generic method
	virtual cmplx getValue(uint i, uint j) const;
	virtual void setValue(uint i, uint j, const cmplx &value);

	virtual uint row() const;
	virtual uint column() const;


	~ColumnDataProxy();
private:
	Domain *m_domain;
};

#endif // COLUMNDATAPROXY_H