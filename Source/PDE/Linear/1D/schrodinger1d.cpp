#include "schrodinger1d.h"

Schrodinger1D::Schrodinger1D(): PDELinear1DVirtual()
{
	dt = 0;
	dx = 0;
	alpha = cmplx(0,0);
}

Schrodinger1D::Schrodinger1D(const Axis *F, int Past, int Future, Type timeStep): PDELinear1DVirtual(Frame(F), Past, Future, cmplx(0,0))
{

	dx = F->getAxisStep();
	dt = timeStep;
	alpha = cmplx(0,dt/(dx*dx));

}

void Schrodinger1D::computeNextStep()
{
	qDebug() << "compute" <<m_Space->getSizeStack();
	LS->SORMethod(m_Space->getCurrentColumn(), m_Space->getNextColumn());
	qDebug() << "Current" << m_Space->getDomain(0);
	qDebug() << "Next" << m_Space->getDomain(-1);
	m_Space->switchDomain();
}

void Schrodinger1D::initializeLinearSolver()
{

	SparseMatrix* M=LS->getSparseMatrix();
	for(int i=1; (i<m_Frame->at(0)->getAxisN()-1); ++i)
	{
		M->setValue(i, i, 1.+alpha);
		M->setValue(i, i-1, -alpha);
		M->setValue(i, i+1, -alpha);
	}
}

void Schrodinger1D::InitialState()
{
	if(!m_Frame)
		return;
	if(!m_Space)
		return;
	if(!LS)
		return;
	DomainBase* D=m_Space->getCurrentDomain();
	for(int i=0; i<m_Frame->at(0)->getAxisN(); ++i)
	{
		Type x= m_Frame->getAxis(0)->getAxisStep()*(Type)i+m_Frame->getAxis(0)->getAxisMin();//Compute true position of x
		cmplx w(0,100.*x);
		cmplx tmp=std::exp(-(x*x)/(Type)4.)*std::exp(w);

		D->setValue(i,tmp);
	}
}
cmplx Schrodinger1D::at(const Point &P) const
{
	//qDebug() << "call at";
	return m_Space->getCurrentDomain()->getValue(P);
}

Schrodinger1D::~Schrodinger1D()
{

}
