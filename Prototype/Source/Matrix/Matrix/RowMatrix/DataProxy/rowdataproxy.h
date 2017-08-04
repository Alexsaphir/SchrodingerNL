#ifndef ROWDATAPROXY_H
#define ROWDATAPROXY_H

#include "../rowmatrixvirtual.h"
#include "../../../../Grid/Base/gridbase.h"

class GridBase;

class RowDataProxy: public RowMatrixVirtual
{
public:
	RowDataProxy();
	RowDataProxy(GridBase *D);
	~RowDataProxy();


	//Specific method for Row or Column Matrix
	cmplx at(int i) const;
	void set(int i, const cmplx &value);

	//Generic method
	cmplx getValue(int i, int j) const;
	void setValue(int i, int j, const cmplx &value);

	int row() const;
	int column() const;

private:
	GridBase *m_domain;
};

#endif // ROWDATAPROXY_H
