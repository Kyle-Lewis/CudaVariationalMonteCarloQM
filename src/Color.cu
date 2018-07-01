#include "VMC.h"

__global__ void Color(float3* colors,
					  float3* rawColorMap,
					  int* rawBinSums,
					  int mapMin,
					  int mapMax,
					  unsigned int simWidth,
					  unsigned int simHeight) // TODO
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = x + simWidth*y;

	if (quadIdx * 4 >= (simWidth * simHeight * 4))
		printf("BAD: Out of colors[] boundary in Color<>, %d , max: %d\n", quadIdx*4, simWidth*simHeight*4);

	int mapped = (int)(0 + (((rawBinSums[quadIdx] - mapMin) * (511 - 0)) / (mapMax - mapMin)));

	for(int i = 0; i < 4; i++)
	{
		if(mapped >= 511)
		{
			colors[4*quadIdx+i] = rawColorMap[510];
		}
		else if(rawBinSums[quadIdx] <= 0)
		{
			colors[4*quadIdx+i] = make_float3(0.15,0.15,0.16);
		}
		else
		{
			colors[4*quadIdx+i] = rawColorMap[mapped];
		}
	}
}