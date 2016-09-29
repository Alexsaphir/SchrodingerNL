#include "include/sparsematrix.h"

SparseMatrix::SparseMatrix() : row(0), column(0)
{
}

SparseMatrix::SparseMatrix(int i, int j) : row(i), column(j)
{

}

cmplx SparseMatrix::getValue(const Position &P) const
{
	return Map.value(P, cmplx(0,0));//Return the value associated with the key. if the value is not find return cmplx(0,0)
}

void SparseMatrix::setValue(const Position &P, const cmplx &v)
{
	Map.insert(P, v);//If P is not in the map add it, else replace only the value
}
