#include <QApplication>
#include <QDebug>

#include "Grid/grid1d.h"
#include "Matrix/SparseMatrix/sparsematrix.h"
#include "Matrix/Matrix/matrix.h"
#include "Matrix/corematrix.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"
#include "Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"

#include "Matrix/MatrixAlgorithm/matrixalgorithm.h"

#include "Solver/Linear/linearsolver.h"

int main(int argc, char **argv)
{
	ColumnMatrix *B = new ColumnMatrix(5);

	ColumnMatrix *X = new ColumnMatrix(5);

	B->set(0,1);//BoundInf
	B->set(2,3);
	B->set(4,1);

	LinearSolver LS(5);
	LS.initSolver(10,.01);

	int nTime(0);
	qDebug() << B->at(0).real() << B->at(1).real() << B->at(2).real() << B->at(3).real() << B->at(4).real();
	while(nTime<=10)
	{
		LS.SORMethod(B,X);
		qDebug() << X->at(0).real() << X->at(1).real() << X->at(2).real() << X->at(3).real() << X->at(4).real();
		nTime++;
		//Permut Pointeur
		ColumnMatrix *tmp;
		tmp=B;
		B=X;
		X=tmp;

	}



	return 0;
}
