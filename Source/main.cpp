#include <QDebug>

#include "Solver/Linear/linearsolver.h"
#include "Axis/linearaxis.h"
#include "Domain/Base/domainbase.h"
#include "Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"
#include "debugclass.h"

int main(int argc, char **argv)
{
	Axis *axe;
	axe = new LinearAxis(-1,1,1);
	Frame *F;
	F = new Frame(axe);

	ColumnMatrix *B;
	ColumnMatrix *X;
	B = new ColumnMatrix(3);
	X = new ColumnMatrix(3);

	DomainBase *B_DB;
	DomainBase *X_DB;
	B_DB = new DomainBase(F, cmplx(0,0));



	LinearSolver *LS;
	LS = new LinearSolver(3);

	SparseMatrix *SM(LS->getSparseMatrix());
	SM->setValue(0, 0, 2);
	SM->setValue(0, 1, -1);
	SM->setValue(1, 0, 1);
	SM->setValue(1, 1, 5);
	SM->setValue(2,2,1);//If we remove this line we can't find a solution
	//The problem in Schrodinger1D it's maybe a bug like this
	SM = NULL;

	B->set(0,3);
	B->set(1,1);

	LS->SORMethod(B, X);

	qDebug() << X->at(0) << X->at(1) << X->at(2);

	delete axe;
	delete F;
	delete B;
	delete X;
	delete LS;
	delete B_DB;
	delete X_DB;

	return 0;
}
