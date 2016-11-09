#include "domainbase.h"

DomainBase::DomainBase(): GridBase(), m_BoundExt(cmplx(0,0))
{

}

DomainBase::DomainBase(const Frame *F, cmplx BoundExt): GridBase(F), m_BoundExt(BoundExt)
{
}

cmplx DomainBase::getValue(const Point &Pos) const
{
	int i = getIndexFromPos(Pos);
	if(i == -1)
		return cmplx(0,0);
	return  GridBase::getValue(i);
}

cmplx DomainBase::getValue(int i) const
{
	if((i<0) || (i>=getSizeOfGrid()))
		return m_BoundExt;
	return  GridBase::getValue(i);
}

DomainBase::~DomainBase()
{

}
