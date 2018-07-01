#include "VMC.h"
__global__ void Zero_Histogram(int* rawBinSums,
							   unsigned int simHeight,
							   unsigned int simWidth)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = x + y * simWidth;
	if (quadIdx >= simHeight * simWidth)
		printf("ERROR: Out of rawBinSums[] bounds, calling: %d, max: %d\n", quadIdx, simHeight*simWidth);

	rawBinSums[quadIdx] = 0;
}