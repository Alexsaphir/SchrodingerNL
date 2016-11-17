#ifndef COLUMNDATAPROXY_H
#define COLUMNDATAPROXY_H

#include "../columnmatrixvirtual.h"
#include "../../../../Domain/Base/domainbase.h"
class DomainBase;

class ColumnDataProxy: public ColumnMatrixVirtual
{
public:
	ColumnDataProxy();
	ColumnDataProxy(DomainBase *D);
	~ColumnDataProxy();

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

#endif // COLUMNDATAPROXY_H
