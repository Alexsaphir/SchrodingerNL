#ifndef DOMAINBASE_H
#define DOMAINBASE_H

#include "../../Grid/Base/gridbase.h"
#include "../../frame.h"
#include "../../type.h"

class DomainBase: public GridBase
{
public:
	DomainBase();
	DomainBase(const Frame *F, cmplx BoundExt);
	virtual ~DomainBase();

	cmplx getValue(const Point &Pos) const;
	virtual cmplx getValue(int i) const;

protected:
	cmplx m_BoundExt;
};

#endif // DOMAINBASE_H
