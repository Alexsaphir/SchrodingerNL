#include "systemevirtual.h"

SystemeVirtual::SystemeVirtual(): m_N(0), m_jacobian(NULL)
{
}

SystemeVirtual::SystemeVirtual(int nbEqua): m_N(nbEqua)
{
	m_V.reserve(m_N);
	m_V.fill(NULL, m_N);
	m_jacobian = new SparseMatrix(m_N, m_N);
}

void SystemeVirtual::addFunction(const FunctionVirtual *F)
{
	m_V.push_back(F->clone());
}

void SystemeVirtual::computeJacobian(CoreMatrix *X)
{
	for(int i=0; i<m_N; ++i)
	{//i: i-equation
		for(int j=0; j<m_N; ++j)
		{//j: j-var
			if(m_V.at(i))
			{
				cmplx tmp;
				tmp = m_V.at(i)->computePartialDerivativeAt(X, j);
				if(tmp != cmplx(0,0))
					m_jacobian->setValue(i, j ,tmp);
			}
		}
	}
}

void SystemeVirtual::computeJacobian(GridBase *G)
{
	this->computeJacobian(G->getColumn());
}

void SystemeVirtual::evaluate(CoreMatrix *X, CoreMatrix *Result) const
{
	if(X == Result)
		return;//Input is the same than the output: Danger!
	if(!X)
		return;//At least one pointer is set to NULL
	if(X->column()*X->row() != m_N)
		return;//there isn't the same number of point
	if(Result->row()*Result->column() != m_N)
		return;
	for(int i=0; i<m_N; ++i)
		Result->set(i, m_V.at(i)->evaluateAt(X));
}

void SystemeVirtual::evaluate(GridBase *G) const
{
	GridBase *temp = new GridBase(*G);
	this->evaluate(temp, G);
	delete temp;
}

void SystemeVirtual::evaluate(GridBase *G, GridBase *Result) const
{
	this->evaluate(G->getColumn(), Result->getColumn());
}

SparseMatrix* SystemeVirtual::getJacobian() const
{
	return m_jacobian;
}

SystemeVirtual::~SystemeVirtual()
{
	delete m_jacobian;
	for(int i=0; i<m_V.size(); ++i)
		delete m_V.at(i);
}
