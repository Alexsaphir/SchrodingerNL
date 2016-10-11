#include <QApplication>
#include <QDebug>

#include "Grid/grid1d.h"
#include "Matrix/SparseMatrix/sparsematrix.h"
#include "Matrix/Matrix/matrix.h"
#include "Matrix/corematrix.h"
#include "Matrix/Matrix/ColumnMatrix/columnmatrix.h"
#include "Matrix/Matrix/ColumnMatrix/DataProxy/columndataproxy.h"

#include "Matrix/MatrixAlgorithm/matrixalgorithm.h"

int main(int argc, char **argv)
{
	SparseMatrix *SparseM = new SparseMatrix(2,2);
	ColumnMatrix *B = new ColumnMatrix(2);

	ColumnMatrix *InitialGuess = new ColumnMatrix(2);

	//x+y
	SparseM->setValue(0,0,1);
	SparseM->setValue(0,1,1);

	//x-y
	SparseM->setValue(1,0,1);
	SparseM->setValue(1,1,-1);

	//x+y=1
	B->set(0,1);
	//x-y=2
	B->set(1,2);

	Type w(.5);

	qDebug() << "\t" << InitialGuess->at(0) <<InitialGuess->at(1);
	//What we do?????

	bool Convergence(false);
	int step(0);
	uint n=SparseM->row();
	while(step!=50)
	{
		for(uint i=0;i<n;++i)
		{
			cmplx sigma(0,0);
			for(uint j=0;j<n;++j)
			{
				if (j!=i)
				{
					sigma+=SparseM->getValue(i,j)*InitialGuess->at(j);
				}
			}
			cmplx Res=B->at(i)-sigma;
			Res/= SparseM->getValue(i,i);
			Res-=InitialGuess->at(i);
			Res*=w;
			Res+=InitialGuess->at(i);
			InitialGuess->set(i, Res);
		}

		qDebug() << "Check convergence"<< step;
		qDebug() << "\t" << InitialGuess->at(0) <<InitialGuess->at(1);
		++step;

	}


	return 0;
}
