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
	QApplication app(argc, argv);


	return app.exec();




}
