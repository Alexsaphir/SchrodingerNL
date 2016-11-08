#ifndef DOMAINPRIVATE_H
#define DOMAINPRIVATE_H

#include "../../Grid/Private/gridprivate.h"
#include "../../frame.h"
#include "../../type.h"

class DomainPrivate: public GridPrivate
{
public:
	DomainPrivate();
	DomainPrivate(const Frame *F, cmplx BoundExt);
	virtual ~DomainPrivate();

	cmplx getValue(const Point &Pos) const;
	virtual cmplx getValue(int i) const;

protected:
	cmplx m_BoundExt;
};

#endif // DOMAINPRIVATE_H
