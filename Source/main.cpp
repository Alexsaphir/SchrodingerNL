#include <QApplication>
#include <QDebug>

#include "Axis/linearaxis.h"
#include "debugclass.h"
#include "Grid/gridmanager.h"
#include "Gui/pdegui1d.h"
#include "Matrix/SparseMatrix/sparsematrix.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"

const Type dt(.001);
const Type dx(.1);
const cmplx alpha(0,dt/dx/dx/2);

void pulse(GridBase *G)
{
	int N=G->getSizeOfGrid();
	const Axis *F=G->getAxis(0);
	for(int i=0; i<N; ++i)
	{
		Type x = F->getAxisMin()+F->getAxisStep()*(Type)(i);
		cmplx w(0,10.*x);
		G->setValue(i,std::exp(-x*x/4.)*std::exp(w));
	}
	G->setValue(0,0.);
	G->setValue(N-1,0.);
}

void f(const GridBase *X, GridBase *Y, GridManager *Data)
{
	//Compute f(X)=Y
	//We assume that the size of X and Y are equal

	const int N(X->getSizeOfAxis(0));
	const cmplx i(0,1);
	//Left-BC
	Y->setValue(0,cmplx(0,0));
	for(int k=1;k<(N-1);++k)
	{
		cmplx tmp(0,0);

		tmp+=1.;
		tmp+=2.*alpha;
		tmp-=i*dt*std::norm(X->getValue(k));
		tmp*=X->getValue(k);

		tmp-=X->getValue(k-1)*alpha;

		tmp-=X->getValue(k+1)*alpha;

		tmp-=Data->getValue(k, 0);//Value at k at the current time

		Y->setValue(k, tmp);
	}

	//Right-BC
	Y->setValue(N-1,cmplx(0,0));
}

void showGrid(GridBase *X , QApplication *app)
{
	PDEGui1D *Fen;
	Fen = new PDEGui1D(X);
	Fen->show();
	app->exec();//wait until Fen is close
	delete Fen;//Fen can be deleted
}

void ComputeJacobian(const GridBase *X,SparseMatrix *Jac, GridManager *Data)
{
	//Compute the jacobian in X
	int N=X->getSizeOfAxis(0);
	for(int i=0; i<N; ++i)
	{
		if((i-1)>=0)
		{
			//Compute derivative in x_(i-1)
			Jac->setValue(i, i-1, -alpha);
		}
		if((i+1)<N)
		{
			//Compute derivative in x_(i+1)
			Jac->setValue(i, i+1, -alpha);
		}
		//Compute derivative in x_i

		cmplx tmp(0,0);
		const cmplx dti(0,dt);
		tmp+=1.;
		tmp+=2.*alpha;
		tmp+=dti*std::norm(X->getValue(i));
		tmp+=-X->getValue(i)*dti*2.*std::conj(X->getValue(i));
		Jac->setValue(i, i, tmp);
	}
}

void LinearSolver(const SparseMatrix *A, ColumnMatrixVirtual *X, ColumnMatrixVirtual *Y, const ColumnMatrixVirtual *B)
{
	//We use Jacobi solver for sparse Matrix
	//X contains the intial guess and store the result at the end
	//Y is a temporary Storage
	//We can modify the value of X and Y because they are given by recopy

	int N=X->row();//Number of row
	int iter(0);
	int iter_max(500);

	while(iter<iter_max)
	{
		//Compute an interation of the solution
		for(int i=0; i<N; ++i)
		{
			cmplx tmp(0.,0.);
			for(int j=0; j<N; ++j)
			{
				if(i!=j)
				{
					tmp-=A->getValue(i, j)*X->at(j);
				}
			}
			tmp += B->at(i);
			tmp=tmp/A->getValue(i,i);
			Y->set(i, tmp);
		}
		//Y contains the new iteration of X

		std::swap(X,Y);
		iter++;
	}
}

void LinearSolverOpti1(const SparseMatrix *A, ColumnMatrixVirtual *X, ColumnMatrixVirtual *Y, const ColumnMatrixVirtual *B)
{
	//We solve Ax=-B
	//We use Jacobi solver for sparse Matrix
	//X contains the intial guess and store the result at the end
	//Y is a temporary Storage
	//We can modify the value of X and Y because they are given by recopy

	int N=X->row();//Number of row
	int iter(0);
	int iter_max(25);
	if(iter_max%2)
		iter_max++;

	while(iter<iter_max)
	{
		//Compute an interation of the solution
#pragma omp parallel for
		for(int i=0; i<N; ++i)
		{
			cmplx tmp(0.,0.);
			for(int j=0; j<N; ++j)
			{
				if(i!=j)
				{
					tmp-=A->getValue(i, j)*X->at(j);
				}
			}
			tmp += -B->at(i);
			tmp=tmp/A->getValue(i,i);
			Y->set(i, tmp);
		}
		//Y contains the new iteration of X
		std::swap(X,Y);
		iter++;
	}
}

void NewtonRaphson(GridManager *Manager)
{
	//We solve directly the next step of the problem
	//We choose the initial guess to be the current step

	GridBase *InitialGuess = Manager->getCurrentGrid();
	GridBase *B = Manager->getTemporaryDomain(0);//We use a grid from the temporary stack
	SparseMatrix *Jac;
	Jac = new SparseMatrix(InitialGuess->getSizeOfGrid(), InitialGuess->getSizeOfGrid());


	GridBase *X;
	GridBase *Xold;
	GridBase *Xtmp;

	Xold = InitialGuess;
	X = Manager->getTemporaryDomain(2);
	Xtmp = Manager->getTemporaryDomain(1);




	*X+=*InitialGuess;
	for(int i=0; i<25; ++i)
	{
		f(Xold, B, Manager);//We have B
		ComputeJacobian(Xold, Jac, Manager);//We have A
		LinearSolverOpti1(Jac, X->getColumn(), Xtmp->getColumn(), B->getColumn());//We can solve the linear system
		//to have X we need to add Xold to X
		*X+=*Xold;

		if(i==0)
			Xold=Manager->getNextGrid();
		std::swap(Xold, X);
		std::swap(X, Xtmp);
	}
	if(Xold != Manager->getNextGrid())
	{
		if(X == Manager->getNextGrid())
		{
			X->reset();
			*X+=*Xold;
		}
		else
		{
			Xtmp->reset();
			*Xtmp+=*Xold;
		}
	}

	//Recopy X in the NextGrid
	delete Jac;

}

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	//Create the Frame with only one dimension
	Axis *X;
	X = new LinearAxis(-5, 5,dx);
	Frame *F;
	F = new Frame(X);

	//Create the GridManager with 1 current time and 1 future time and 3 temporary Grid
	GridManager *Manager;
	Manager = new GridManager(0,1,3,*F);
	const int N(Manager->getCurrentGrid()->getSizeOfGrid());
	qDebug() << alpha << N;
	//Init the Current Grid with the pulse
	pulse(Manager->getCurrentGrid());

	//Show the initial state aka the current grid
	showGrid(Manager->getCurrentGrid(), &app);


	for(int i=0; i<1000; ++i)
	{
		NewtonRaphson(Manager);
		Manager->switchGrid();
		if((i%10)==0)
			qDebug() << i;
	}
	showGrid(Manager->getCurrentGrid(), &app);



	delete Manager;
	delete F;
	delete X;

	return 0;
	//return app.exec();
}
