#include "KernelUtility.cuh"

__device__ int getGlobalIndex_2D_2D()
{
	int BlockId = blockIdx.x + blockIdx.y*gridDim.x;
	return BlockId*(blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;//ThreadId
}


int KernelUtility::computeNumberOfBlocks(int nbThreadPerBlock, int nbThread)
{
	return ((nbThread % nbThreadPerBlock) == 0) ? (nbThread / nbThreadPerBlock) : (1 + nbThread / nbThreadPerBlock);
}

int KernelUtility::computeNumberOfBlocks(int nbThread)
{
	return KernelUtility::computeNumberOfBlocks(1024, nbThread);
}

dim3 KernelUtility::computeNumberOfBlocks2D(dim3 blockSize, int Nx, int Ny)
{
	dim3 gridSize;
	gridSize.x = computeNumberOfBlocks(blockSize.x, Nx);
	gridSize.y = computeNumberOfBlocks(blockSize.y, Ny);
	return gridSize;
}
