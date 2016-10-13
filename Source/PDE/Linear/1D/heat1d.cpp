#include "heat1d.h"

Heat1D::Heat1D(): PDELinear1DVirtual(), Grid1(NULL), Grid2(NULL), Grid1IsCurrent(true)
{

}
Heat1D::Heat1D(const Axis &X, Type t, cmplx Binf, cmplx Bsup): PDELinear1DVirtual()
{
	Grid1 = new Grid1D(X);
	Grid2 = new Grid1D(X);

	BoundInf = Binf;
	BoundSup = Bsup;

	dt = t;
	dx=X.getAxisStep();

	Grid1IsCurrent =true;

	LS = new LinearSolver(X.getAxisN());

	//C1 =new ColumnDataProxy()
}

void Heat1D::initMatrix()
{
//	SparseMatrix *M = LS->getSparseMatrix();

//	if(!M)
//		return;
//	Type alpha=dt/dx/dx;
//	M->setValue(0, 0, BoundInf);//Boundary inf condition
//	for(uint i(1); i<(M->row()-1); ++i)
//	{
//		M->setValue(i, i, 1.+2.*alpha);
//		M->setValue(i, i-1, -alpha);//uint => 0-1==2^32
//		M->setValue(i, i+1, -alpha);//if out of order, it's catch by the sparseMatrix
//	}
//	M->setValue(M->row()-1, M->row()-1, BoundSup);//Boundary sup condition

	LS->initSolver(dt, dx);;
}

void Heat1D::compute()
{
	if(Grid1IsCurrent)
	{
		LS->SORMethod(Grid1,Grid2);
		Grid1IsCurrent = false;
		return;
	}
	else
	{

		LS->SORMethod(Grid2,Grid1);
		Grid1IsCurrent = true;
		return;
	}
}

void Heat1D::pulse()
{
	for(uint i(1); i<(Grid1->getN()-1); ++i)
	{
		Type x = Grid1->getXmin()+(Type)(i)*dx;
		Grid1->setValue(i,std::exp(-x*x/2));
	}
	Grid1IsCurrent =true;
}

cmplx Heat1D::get(uint i) const
{
	Grid1D *G;
	if(Grid1IsCurrent)
		G=Grid1;
	else
		G=Grid2;
	return G->getValue(i);
}

Heat1D::~Heat1D()
{
	if(Grid1)
		delete Grid1;
	if(Grid2)
		delete Grid2;
}
