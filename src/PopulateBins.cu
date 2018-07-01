#include "VMC.h"

	/* Populate the bins of the histogram for the simulation, 
	 * just for visualization purposes */

__global__ void PopulateBins(int* rawBinSums,
							 float2* rawRingPoints,
							 float xRange,
							 float yRange,
							 int mode, 
							 int numPoints,
							 unsigned int simHeight,
							 unsigned int simWidth)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int xBin = 0;
	unsigned int yBin = 0;

	while (rawRingPoints[idx].x > (-xRange / 2.0 + xBin * xRange / simWidth)) xBin++;
	while (rawRingPoints[idx].y > (-yRange / 2.0 + yBin * yRange / simHeight)) yBin++;

	if (idx == 10)
	{
		printf("Idx 10 reads X: %f, Y: %f\n", rawRingPoints[idx].x, rawRingPoints[idx].y);
		printf("Idx 10 trying to add to bin: %d, %d, %d\n", xBin, yBin, xBin + yBin * simWidth);
	}

	if(xBin > simWidth - 1 || yBin > simHeight - 1){return;};

	atomicAdd(&rawBinSums[xBin + yBin * simWidth], 1);
}