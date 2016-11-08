#ifndef DOMAIN_H
#define DOMAIN_H

#include "Private/domainprivate.h"

class Domain: public DomainPrivate
{
public:
	Domain();
	Domain(const Frame &F, cmplx BoundExt);
	Domain(const Axis *X, cmplx BoundExt);
	Domain(const Axis *X, const Axis *Y, cmplx BoundExt);
	Domain(const Domain &D);
	virtual ~Domain();
private:
	cmplx getBoundaryCondition(const Point &Pos) const;
protected:
	const Frame *m_Frame;
};

#endif // DOMAIN_H
