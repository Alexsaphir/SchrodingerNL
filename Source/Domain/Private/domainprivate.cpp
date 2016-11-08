#include "domainprivate.h"

DomainPrivate::DomainPrivate(): GridPrivate(), m_BoundExt(cmplx(0,0))
{

}

DomainPrivate::DomainPrivate(const Frame *F, cmplx BoundExt): GridPrivate(F), m_BoundExt(BoundExt)
{
}

cmplx DomainPrivate::getValue(const Point &Pos) const
{
	int i = getIndexFromPos(Pos);
	if(i == -1)
		return cmplx(0,0);
	return  GridPrivate::getValue(i);
}

cmplx DomainPrivate::getValue(int i) const
{
	if((i<0) || (i>=getSizeOfGrid()))
		return m_BoundExt;
	return  GridPrivate::getValue(i);
}

DomainPrivate::~DomainPrivate()
{

}
