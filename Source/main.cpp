#include <QApplication>
#include <QDebug>

#include "Grid/grid1d.h"
#include "Matrix/SparseMatrix/sparsematrix.h"
#include "Matrix/Matrix/matrix.h"
#include "Matrix/corematrix.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"
#include "Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"

int main(int argc, char **argv)
{
	Grid1D G(Axis(-1,1,1));
	Grid1D R(Axis(-1,1,1));
	Domain *D = new Domain(cmplx(0,0));
	D->AddAxis(Axis(-1,1,1));
	D->AddAxis(Axis(-1,.1,1));
	D->initGrid();

	QVector<ColumnMatrixVirtual*> CM;
	//CM.push_back(new SparseMatrix(3,3));
	//CM.push_back(new Matrix(3,3));

	CM.push_back(new ColumnDataProxy(D));



	CM.at(0)->setValue(0,1,cmplx(.2,1));
	//CM.at(1)->setValue(0,0,cmplx(10,10));



	qDebug().noquote() << CM.at(0)->at(0) << CM.at(0)->getValue(0, 1);

	delete CM.at(0);
//	delete CM.at(1);

	return 0;
}
