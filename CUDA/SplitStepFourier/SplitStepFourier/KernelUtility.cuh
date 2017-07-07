#pragma once

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

namespace KernelUtility
{
	int computeNumberOfBlocks(int nbThreadPerBlock, int nbThread);
	int computeNumberOfBlocks(int nbThread);
}

