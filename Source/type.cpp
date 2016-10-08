#include "type.h"

QDebug operator<< (QDebug dbg, const cmplx &z)
{

	QString s1,s2;
	s1.setNum(z.real());
	s2.setNum(z.imag());
	dbg << s1 + "+" + s2 + "i";
	return dbg;
}
