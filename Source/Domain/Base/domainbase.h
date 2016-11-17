#ifndef DOMAINBASE_H
#define DOMAINBASE_H

#include "../../Grid/Base/gridbase.h"
#include "../../Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"
#include "../../Matrix/Matrix/RowMatrix/DataProxy/rowdataproxy.h"

class RowDataProxy;
class ColumnDataProxy;

class DomainBase: public GridBase
{
public:
	DomainBase();
	DomainBase(const Frame *F, cmplx BoundExt);
	virtual ~DomainBase();

	cmplx getValue(const Point &Pos) const;
	virtual cmplx getValue(int i) const;

	ColumnDataProxy* getColumn() const;
	RowDataProxy* getRow() const;

protected:
	cmplx m_BoundExt;
private:
	ColumnDataProxy *m_ProxyColumn;
	RowDataProxy *m_ProxyRow;
};

#endif // DOMAINBASE_H
