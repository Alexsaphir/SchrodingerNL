#ifndef SYSTEMEVIRTUAL_H
#define SYSTEMEVIRTUAL_H

#include <QVector>

#include "../Function/functionvirtual.h"

class SystemeVirtual
{
public:
	SystemeVirtual();
	virtual ~SystemeVirtual();


private:
	int m_N;//Number of equations
	QVector<FunctionVirtual*> m_V;
};

#endif // SYSTEMEVIRTUAL_H
