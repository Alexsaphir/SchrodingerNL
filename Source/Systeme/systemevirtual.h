#ifndef SYSTEMEVIRTUAL_H
#define SYSTEMEVIRTUAL_H

#include <QVector>

#include "../Function/functionvirtual.h"

#include "../Matrix/SparseMatrix/sparsematrix.h"

class SystemeVirtual
{
public:
	SystemeVirtual();
	SystemeVirtual(int nbEqua);
	virtual ~SystemeVirtual();

	void addFunction(const FunctionVirtual *F);

	virtual void computeJacobian(CoreMatrix *X);
	virtual void computeJacobian(GridBase *G);

	SparseMatrix* getJacobian() const;

	void evaluate(CoreMatrix *X, CoreMatrix *Result) const;
	void evaluate(GridBase *G) const;
	void evaluate(GridBase *G, GridBase *Result) const;


private:
	int m_N;//Number of equations
	SparseMatrix *m_jacobian;
	QVector<FunctionVirtual*> m_V;
};

#endif // SYSTEMEVIRTUAL_H

//It's this class who managed the pointer in m_V
