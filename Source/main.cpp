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
#include "SolverGui/pdegui1d.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	PDEGui1D *F;
	F= new PDEGui1D();
	F->show();
	F->refreshView();


	return app.exec();




}
