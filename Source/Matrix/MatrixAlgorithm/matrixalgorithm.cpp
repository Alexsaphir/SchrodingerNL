#include "matrixalgorithm.h"

MatrixAlgorithm::MatrixAlgorithm()
{

}

void MatrixAlgorithm::MatrixAddition(const CoreMatrix *A, const CoreMatrix *B, CoreMatrix *C)
{
	if(!A || !B || !C)
		return;//Null Matrix
	if(A->row()!=B->row() || A->row()!=C->row())
		return;
	if(A->column()!=B->column()|| A->column()!=C->column())
		return;

	for(int i(0); i<A->row(); ++i)
	{
		for(int j(0); j<A->column(); ++j)
		{
			C->setValue(i, j, A->getValue(i, j)+B->getValue(i, j));
		}
	}
}

void MatrixAlgorithm::MatrixScalarMultiplication(cmplx s, const CoreMatrix *A, CoreMatrix *C)
{
	if(!A || !C)
		return;
	if(A->row()!=C->row())
		return;
	if(A->column()!=C->column())
		return;

	for(int i(0); i<A->row(); ++i)
	{
		for(int j(0); j<A->column(); ++j)
		{
			C->setValue(i,j,s*A->getValue(i, j));
		}
	}
}

void MatrixAlgorithm::MatrixTranspose(const CoreMatrix *A, CoreMatrix *C)
{
	if(!A || !C)
		return;
	if(A->row()!=C->column())
		return;
	if(A->column()!=C->row())
		return;
	for(int i(0); i<A->row(); ++i)
	{
		for(int j(0); j<A->column(); ++j)
		{
			C->setValue(j,i,A->getValue(i, j));
		}
	}
}

void MatrixAlgorithm::MatrixMultiplication(const CoreMatrix *A, const CoreMatrix *B, CoreMatrix *C)
{
	if(!A || !B || !C)
		return;//Null Matrix
	if(A->column()!=B->row())
		return;
	if(A->row()!=C->row())
		return;
	if(B->column()!=C->column())
		return;

	for(int i(0); i<C->row(); ++i)
	{
		for(int j(0); j<C->column(); ++j)
		{
			cmplx tmp(0,0);
			for(int k(0); k<A->column(); ++k)
			{
				tmp+=A->getValue(i, k)*B->getValue(k, j);
			}

			C->setValue(i, j, tmp);
		}
	}
}
