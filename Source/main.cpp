#include <QApplication>
#include <QDebug>
#include <iostream>

#include "Axis/linearaxis.h"
#include "debugclass.h"
#include "Grid/gridmanager.h"
#include "Gui/pdegui1d.h"
#include "Matrix/SparseMatrix/sparsematrix.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"

const Type dt(.01);
const Type dx(.1);

const Type LAMBDA(-1.);

cmplx iMul(cmplx z)
{
	return z*cmplx(0,1);
}


void showGrid(GridBase *X , QApplication *app)
{
	PDEGui1D *Fen;
	Fen = new PDEGui1D(X);
	Fen->show();
	app->exec();//wait until Fen is close
	delete Fen;//Fen can be deleted
}


void pulse(GridBase *G)
{
	int N=G->getSizeOfGrid();
	const Axis *F=G->getAxis(0);
	for(int i=0; i<N; ++i)
	{
		Type x = F->getAxisMin()+F->getAxisStep()*(Type)(i);
		//cmplx w(0,10.*x);

		cmplx w(0,10.*x);
		G->setValue(i,2*std::exp(-x*x/9.)*std::exp(w));
	}
	G->setValue(0,0.);
	G->setValue(N-1,0.);
}

void initPhi(const GridBase *V, GridBase *Phi)
{
#pragma omp parallel for
	for(int i=0;i<V->getSizeOfGrid(); ++i)
	{
		Phi->setValue(i,std::norm(V->getValue(i)));
	}
}

void evaluatePhi(const GridBase *V, GridBase *Phi)
{
#pragma omp parallel for
	for(int i=0;i<V->getSizeOfGrid(); ++i)
	{
		Phi->setValue(i,2.*std::norm(V->getValue(i))-Phi->getValue(i));
	}
}

void evaluateB(const GridBase *V, const GridBase *Phi, GridBase *B)
{
#pragma omp parallel for
	for(int i=1;i<V->getSizeOfGrid()-1; ++i)
	{
		B->setValue(i,LAMBDA*Phi->getValue(i)*V->getValue(i)/2.
					+ iMul(V->getValue(i))/dt
					- .25*(V->getValue(i-1)-2.*V->getValue(i)+V->getValue(i+1))
					);
	}
	int i=0;
	B->setValue(i,LAMBDA*Phi->getValue(i)*V->getValue(i)/2.
				+ iMul(V->getValue(i))/dt
				- .25*(-2.*V->getValue(i)+V->getValue(i+1))
				);
	i=V->getSizeOfGrid()-1;
	B->setValue(i,LAMBDA*Phi->getValue(i)*V->getValue(i)/2.
				+ iMul(V->getValue(i))/dt
				- .25*(V->getValue(i-1)-2.*V->getValue(i))
				);
}

void computeSparseMatrix(SparseMatrix* M, const GridBase *V, const GridBase *Phi)
{
	const cmplx beta=1./4./dx/dx;
	const cmplx alpha=iMul(1./dt)-1./2./dx/dx;
#pragma omp parallel for
	for(int i=0;i<V->getSizeOfGrid(); ++i)
	{
		M->setValue(i,i, alpha+.5*LAMBDA*Phi->getValue(i));
		M->setValue(i,i+1,beta);
		M->setValue(i,i-1,beta);
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
	int iter_max(150);

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
			tmp += B->at(i);
			tmp=tmp/A->getValue(i,i);
			Y->set(i, tmp);
		}
#pragma omp parallel for
		for(int i=0; i<N; ++i)
		{
			cmplx tmp(0.,0.);
			for(int j=0; j<N; ++j)
			{
				if(i!=j)
				{
					tmp-=A->getValue(i, j)*Y->at(j);
				}
			}
			tmp += B->at(i);
			tmp=tmp/A->getValue(i,i);
			X->set(i, tmp);
		}

		iter+=2;
	}
}



int main(int argc, char **argv)
{
	QApplication app(argc, argv);



	//Create the Frame with only one dimension
	Axis *X;

	X = new LinearAxis(-40, 40,dx);
	qDebug() << "Spatial resolution : " << X->getAxisN() << X->getAxisStep();
	Frame *F;
	F = new Frame(X);


	//Create the GridManager with 1 current timeand 3 temporary Grid
	GridManager *Manager;
	Manager = new GridManager(0,0,3,*F);
	const int N(Manager->getCurrentGrid()->getSizeOfGrid());

	GridBase *B=Manager->getTemporaryDomain(0);
	GridBase *Phi=Manager->getTemporaryDomain(1);
	GridBase *Vtmp=Manager->getTemporaryDomain(2);

	GridBase *V=Manager->getCurrentGrid();

	SparseMatrix *M =new SparseMatrix(N,N);



	//Init the Current Grid with the pulse
	pulse(V);
	initPhi(V, Phi);

	//Show the initial state aka the current grid
	showGrid(Manager->getCurrentGrid(), &app);


	for(int i=0; i<100;++i)
	{
		computeSparseMatrix(M, V, Phi);
		evaluateB(V, Phi, B);
		LinearSolver(M,V->getColumn(),Vtmp->getColumn(),B->getColumn());

		evaluatePhi(V, Phi);
		//if(i%10==0)
			qDebug() << i*1<<"%";
		//showGrid(Manager->getCurrentGrid(), &app);
	}
	showGrid(Manager->getCurrentGrid(), &app);

	delete M;
	delete Manager;
	delete F;
	delete X;

	return 0;
	//return app.exec();
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

