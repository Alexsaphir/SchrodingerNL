#include "debugclass.h"

QDebug operator<< (QDebug dbg, const cmplx &z)
{
	QString s1,s2;
	s1.setNum(z.real());
	s2.setNum(z.imag());
	dbg << s1 + "+" + s2 + "i";
	return dbg;
}

QDebug operator<< (QDebug dbg, const Axis &z)
{
	dbg << "[" << QString::number(z.getAxisMin()) << ":" << QString::number(z.getAxisStep()) << ":" << QString::number(z.getAxisMax()) << "]";
	return dbg;
}

QDebug operator<< (QDebug dbg, const CoreMatrix &M)
{
	//dbg << " ";
	for(int i=0; i<M.row(); ++i)
	{
		for(int j=0; j<M.column(); ++j)
		{
			dbg << std::norm(M.getValue(i,j)) ;
		}

		dbg << "\n";
	}
	return dbg;
}
