#include <stdio.h>
__global__ void FormPNGData(float3* colors,
						    unsigned char* pixelData, 
						    unsigned int simWidth, 
						    unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = x + simWidth*y;

	if (quadIdx * 4 >= (simWidth * simHeight * 4))
		printf("BAD: Out of colors[] boundary in FormPNGData, %d , max: %d\n", quadIdx*4, simWidth*simHeight*4);
	if (quadIdx*3 + 2 >= (simWidth * simHeight * 3))
		printf("BAD: Out of pixelData[] boundary in FormPNGData, %d, max: %d\n", quadIdx*3+2, simWidth*simHeight*3);
	

		// scale and map floating point pixel data [0.0,1.0] to unsigned char* 
		// data in [0,255] for each RGB value. Size of device data 
		// should match (3 * float3's == 4 * num pixelData)
	unsigned char r = (unsigned char)(colors[4 * quadIdx].x * (255));
	unsigned char g = (unsigned char)(colors[4 * quadIdx].y * (255));
	unsigned char b = (unsigned char)(colors[4 * quadIdx].z * (255));
	
	pixelData[3 * quadIdx + 0] = r;
	pixelData[3 * quadIdx + 1] = g;
	pixelData[3 * quadIdx + 2] = b;
}