#include "matrixalgorithm.h"

MatrixAlgorithm::MatrixAlgorithm()
{

}

MatrixAlgorithm::MatrixAddition(const CoreMatrix *A, const CoreMatrix *B, CoreMatrix *C)
{
	if(!A || !B || !C)
		return;//Null Matrix
	if(A->row()!=B->row() || A->row()!=C->row())
		return;
	if(A->column()!=B->column()|| A!=C->column())
		return;
	for(uint i(0); i<A->row(); ++i)
	{
		for(uint j(0); j<A->column(); ++j)
		{
			C->setValue(i,j, A->getValue(i,j)+B->getValue(i,j));
		}
	}
}
