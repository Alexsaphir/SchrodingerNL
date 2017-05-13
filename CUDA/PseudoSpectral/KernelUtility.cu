#include "KernelUtility.cuh"

int KernelUtility::computeNumberOfBlocks(int nbThreadPerBlock, int nbThread)
{
	return ((nbThread % nbThreadPerBlock) == 0) ? (nbThread / nbThreadPerBlock) : (1 + nbThread / nbThreadPerBlock);
}

int KernelUtility::computeNumberOfBlocks(int nbThread)
{
	return KernelUtility::computeNumberOfBlocks(1024, nbThread);
}
