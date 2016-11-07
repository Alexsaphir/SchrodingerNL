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

}

void Schrodinger1D::initializeLinearSolver()
{

	SparseMatrix* M=LS->getSparseMatrix();
	for(int i=1; (i<Repere->size()-1); ++i)
	{
		M->setValue(i, i, 1.+alpha);
		M->setValue(i, i-1, -alpha);
		M->setValue(i, i+1, -alpha);
	}
}

void Schrodinger1D::InitialState()
{
	if(!Repere)
		return;
	if(!Space)
		return;
	if(!LS)
		return;
	Domain* D=Space->getCurrentDomain();
	for(int i=0; i<Repere->size(); ++i)
	{
		Type x= Repere->getAxis(0)->getAxisStep()*(Type)i+Repere->getAxis(0)->getAxisMin();//Compute true position of x


		cmplx w(0,100.*x);
		cmplx tmp=std::exp(-(x*x)/(Type)4.)*std::exp(w);

		D->setValue(i,tmp);
	}
}

Schrodinger1D::~Schrodinger1D()
{

}
