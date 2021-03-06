#include "sparsematrix.h"

SparseMatrix::SparseMatrix(int row, int column): CoreMatrix(row, column)
{
	for(int i(0); i<row; ++i)
	{
		V.push_back(new QMap<int,cmplx>);
	}
}

cmplx SparseMatrix::getValue(int i, int j) const
{
	if(i>=m_row)
		return cmplx(0,0);
	return V.at(i)->value(j, cmplx(0,0));//Return the value associated with the key. if the value is not find return cmplx(0,0)
}

void SparseMatrix::setValue(int i, int j, const cmplx &value)
{
	if (i>=m_row || j>=m_column)
		return;//Index out of range
	V.at(i)->insert(j, value);//If P is not in the map add it, else replace only the value
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
	for(int i(0); i<m_row;++i)
	{
		cmplx tmpR(0);
		QMapIterator<int, cmplx> iterator(*V.at(i));
		while (iterator.hasNext())
		{
			iterator.next();
			int j(iterator.key());
			tmpR+=iterator.value()*S->getValue(j);
		}
		R->setValue(i,tmpR);
	}
}

SparseMatrix::~SparseMatrix()
{
	for(int i(0); i<m_row; ++i)
	{
		delete V.at(i);
	}
}
