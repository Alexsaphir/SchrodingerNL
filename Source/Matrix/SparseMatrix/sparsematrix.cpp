#include "sparsematrix.h"

SparseMatrix::SparseMatrix(uint row, uint column): CoreMatrix(row, column)
{

}

cmplx SparseMatrix::getValue(uint i, uint j) const
{
	return Map.value(QPair<int,int>(i,j), cmplx(0,0));//Return the value associated with the key. if the value is not find return cmplx(0,0)
}

void SparseMatrix::setValue(uint i, uint j, const cmplx &value)
{
	if (i>=m_row || j>=m_column)
		return;//Index out of range
	Map.insert(QPair<int,int>(i,j), value);//If P is not in the map add it, else replace only the value
}

void SparseMatrix::dotByGrid1D(Grid1D *S, Grid1D *R)
{
	/*We suppose that the dot product is realizable
	 * So S.size()==G.size()
	 * S is a square matrix
	 */

	//Put all value of R at 0
	for (int i=0; i<R->getN(); ++i)
	{
		R->setValue(i,cmplx(0,0));
	}

	//Begin calculus
	QMapIterator<Position, cmplx> iterator(Map);
	while (iterator.hasNext())
	{

		iterator.next();
		qDebug() << iterator.key() << ": " << iterator.value().real();
		int i=iterator.key().first;
		R->setValue(i, R->getValue(i)+S->getValue(iterator.key().second)*iterator.value());
	}
}

SparseMatrix::~SparseMatrix()
{

}
